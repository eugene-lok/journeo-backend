from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.models import UserItinerary, UserMessage
from app.middleware import addCorsMiddleware
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from pydantic import BaseModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from app.mapboxRoutes import getRouteFromMapbox
from app.loggerConfig import logger
from app.geoTools.geocoding import *
from app.extractorAgent import createTravelPreferenceWorkflow
#from app.auth import authRouter
import os
import re
import httpx
import asyncio
import json

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)



responseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "itinerary",
        "schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "integer"},
                            "places": {
                                "type": "array",
                                "items": {  
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" },
                                        "address": { "type": "string" },
                                        "description": { "type": "string" },
                                    },
                                    "required": ["name", "address", "description"],
                                    "additionalProperties": False
                                }
                            },
                            "summaryOfDay": {"type": "string"}
                        },
                        "required": ["day","places","summaryOfDay"],
                        "additionalProperties": False                       
                    }              
                }
            },
            "required": ["days"],
            "additionalProperties": False
        },
        "strict": True,
    }
}

app = FastAPI()

# Apply CORS middleware
addCorsMiddleware(app)

# Add router
#app.include_router(authRouter)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1500,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
    response_format=responseFormat
)


## Put in options for adults and kids
## Implement different budget options, don't have to ask for budget

systemPromptShort = (
    f"You are a travel agent. Your job is to generate a complete itinerary based on the user’s input. "
    f"Your goal is to gather the following information from the user:\n"
    f"- **Trip duration** (phrased as 'X days', 'X-day trip', 'a week', or 'from date A to date B').\n"
    f"- **Trip origin** (phrased as 'from X', 'starting in X', or 'origin is X').\n"
    f"- **Trip destination** (phrased as 'to X', 'destination is X', or 'visit X').\n"
    f"- **Number of travellers** (phrased as 'for X people', 'with X people', 'X people going', 'alone', or 'solo').\n"
    f"- **Trip budget** (phrased as '$X', 'a budget of X', or 'around X').\n\n"

    f"### Duration Handling:\n"
    f"**If the user specifies the duration in any form (e.g., 'X days', 'a week', or a range like 'from date A to date B'), assume the duration is complete and do not ask for it again.**\n\n"

    f"### Origin Handling:\n"
    f"**If the user specifies the origin with phrases like 'from X', 'starting in X', or 'origin is X', assume the origin is complete and do not ask for it again.**\n\n"

    f"### Destination Handling:\n"
    f"**If the user specifies the destination with phrases like 'to X', 'destination is X', or 'visit X', assume the destination is complete and do not ask for it again.**\n\n"

    f"### Number of Travellers Handling:\n"
    f"**If the user specifies the number of travellers with phrases like 'for X people', 'with X people', 'X people going', 'alone', or 'solo', assume the number of travellers is complete and do not ask for it again. Interpret 'alone' or 'solo' as 1 traveller.**\n\n"
)

systemMessage = SystemMessagePromptTemplate.from_template(systemPromptShort)
messageHistory = MessagesPlaceholder(variable_name="messages")
messagesList = []
class UserInputModel(BaseModel):
    user_input: str

workflow = createTravelPreferenceWorkflow()   

@app.post("/api/extract-preferences/")
async def extract_travel_preferences(input_data: UserInputModel):
    print(input_data)
    try:
        # Configuration for the workflow
        config = {"configurable": {"thread_id": "preference_extraction"}}
        
        # Prepare initial input
        initial_input = {
            'userInput': input_data.user_input,
            'previousEntities': None
        }
        
        print(f"userInput: {initial_input}")
        # Run the workflow
        result = workflow.invoke(initial_input, config=config)
        
        # Return the extracted entities and missing entities
        return {
            "extracted_entities": result.get('extractedEntities', {}),
            "missing_entities": result.get('missingEntities', []),
            "is_complete": result.get('isComplete', False),
            "clarificationMessage": result.get('clarificationMessage', "")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/")
async def chatResponse(message: UserMessage):

    try:
        humanMessage = HumanMessagePromptTemplate.from_template("{input}")
        
        prompt = ChatPromptTemplate.from_messages(
            [
                systemMessage, 
                humanMessage,
                messageHistory           
            ]
        )

        # Add human message in message list
        messagesList.append(HumanMessage(content=message.input))

        # Generate the response using the chain
        chain = prompt | llm

        response = chain.invoke(
            {
                "input": message.input, 
                "messages": messagesList
            }
        )
        
        # Validate response
        if not response.content or not response.content.strip():
            return None, "Error: Empty or null JSON response."
        try:
            itineraryContent = json.loads(response.content)
            print(f"\nResponse Formatted: {response.content}")
            # Check for incomplete structure
            for day in itineraryContent.get("days"):
                if "day" not in day or "places" not in day or not isinstance(day["places"], list):
                    return None, "Error: Missing or invalid structure in 'days' element."
                else:
                    continue

            # Store response in message list
            messagesList.append(AIMessage(content=response.content))
            
            names = []
            addresses = []
            for day in itineraryContent["days"]:
                for place in day["places"]:
                    names.append(place["name"])
                    addresses.append(place["address"])

            # Get coordinates and googlePlaceId of addresses
            placesInfo = await getAllPlaceDetails(names, addresses)

            places = []

            # Assign attributes to each place
            for id, (name, address, placeInfo) in enumerate(zip(names, addresses, placesInfo)):
                place = {
                    "id": id,
                    "name": name,
                    "isAirport": None,
                    "address": address,
                    "initialPlaceId": placeInfo['initialPlaceId'],
                    "coordinates": placeInfo['coordinates'],
                    "predictedLocation": placeInfo['placePrediction'],
                    "details": placeInfo['details']
                }
                checkIfAirport(place)
                places.append(place)
            
            # TODO: # Create new keys in itinerary for place details
            # Prepare response data
            responseData = {
                "itinerary": itineraryContent,
                "places": places
            }
            # TODO: # Organize routes by day
            routes = []

            if len(places) >= 2:
                # List of consecutive place pairs
                consecutivePairs = []
                for i in range(len(places) - 1):
                    fromPlace = places[i]
                    toPlace = places[i + 1]
                    consecutivePairs.append((fromPlace, toPlace))

                # Filter out pairs where both are airports
                nonAirportPairs = []
                for fromPlace, toPlace in consecutivePairs:
                    if not (fromPlace.get("isAirport", False) and toPlace.get("isAirport", False)):
                        nonAirportPairs.append((fromPlace, toPlace))

                # Create route fetching tasks and keep track of pairs
                routeTasks = []
                routePairs = []

                async with httpx.AsyncClient() as client:
                    for fromPlace, toPlace in nonAirportPairs:
                        task = getRouteFromMapbox(
                            client,
                            startCoords=fromPlace["coordinates"],
                            endCoords=toPlace["coordinates"]
                        )
                        routeTasks.append(task)
                        routePairs.append((fromPlace, toPlace))

                    # Gather all route tasks
                    route_results = await asyncio.gather(*routeTasks, return_exceptions=True)

                # Add route pairs and routes
                for index, result in enumerate(route_results):
                    fromPlace, toPlace = routePairs[index]

                    # Check for exception
                    if isinstance(result, Exception):
                        print(f"Failed to fetch route between {fromPlace['name']} and {toPlace['name']}: {result}")
                        continue 
                    
                    # Append routes to list
                    if result:
                        routes.append({
                            "from": {
                                "id": fromPlace["id"],
                                "name": fromPlace["name"],
                                "coordinates": fromPlace["coordinates"]
                            },
                            "to": {
                                "id": toPlace["id"],
                                "name": toPlace["name"],
                                "coordinates": toPlace["coordinates"]
                            },
                            "route": result 
                        })
                    else:
                        print(f"No route found between {fromPlace['name']} and {toPlace['name']}.")

            # Add routes to response 
            responseData["routes"] = routes

            #print(responseData)
            # Return formatted response
            botResponse = {"response": responseData}
            return JSONResponse(content=botResponse)        

        except json.JSONDecodeError as e:
            return None, f"Error: Failed to decode JSON. {str(e)}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Get place type and details from Google Text Search API
async def getPlaceDetailsFromText(client, textQuery):
    googleAPIKey = os.getenv("GOOGLE_API_KEY")
    # Field mask
    fields = "places.id,places.displayName,places.formattedAddress,places.primaryType,places.googleMapsUri,places.websiteUri,places.rating,places.photos"
    # Construct request params and body
    textSearchUrl = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': googleAPIKey,
        'X-Goog-FieldMask': fields
    }
    body = {
        "textQuery": textQuery,
    }
    try:
        response = await client.post(textSearchUrl, headers=headers, json=body)
        if response.status_code == 200:
            data = response.json()
            #print(data)
            if 'places' in data:
                places = data['places']
                results = []
                # Map data
                for place in places:
                    results.append({
                        "googlePlaceId": place["id"],
                        "displayName": place["displayName"],
                        "formattedAddress": place["formattedAddress"],
                        "primaryType": place["primaryType"],
                        "googleMapsUri": place["googleMapsUri"],
                        "websiteUri": place["websiteUri"],
                        "rating": place["rating"],
                        #"photos": places['photos']
                    })
                print(f"results: {results}")
                return results
            else:
                print(f"No places found for query: {textQuery}")
                return None
        else:
            print(f"Error fetching places for query '{textQuery}': {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching places for query '{textQuery}': {e}")
        return None

# Get GeoJSON routes from Mapbox Directions API

# Get lat, long from Mapbox Geocoding API 
""" async def getCoordinates(client, address):
    print(f"Address: {address}")
    accessToken = os.getenv("MAPBOX_ACCESS_TOKEN")
    geocodeUrl = f"https://api.mapbox.com/search/geocode/v6/forward?q={address}&access_token={accessToken}"
    response = await client.get(geocodeUrl)
    if response.status_code == 200:
        data = response.json()
        if data["features"]:
            coords = data["features"][0]["geometry"]["coordinates"]
            return {"latitude": coords[1], "longitude": coords[0]}
        return None """


# Generates itinerary and fetches coordinates
"""
@app.post("/api/generateItinerary/")
async def generateItinerary(userItinerary: UserItinerary):
    # Prepare the prompt for the OpenAI API
    client = OpenAI()

    systemPrompt = (
        f"You are a helpful travel agent. You will be generating a highly specific itinerary based on parameters provided by the user. "
        f"The itinerary will be organized per day. Do not include any destinations in the origin, but dedicate the first and last days of "
        f"the itinerary to travel from origin to first the destination, and the final destination back to the origin, respectively. "
        f"For each location you recommend to visit, the following must be included in **separate lines**: "
        f"- **Name** of the location (always start with Name:)."
        f"- **Address** for geocoding."
        f"- **Description** of the location."
        f"After the itinerary itself, include a section for a budget breakdown. Ensure the following formatting rules: "
        f"- The keywords **Name:**, **Address:**, and **Description:** must **always be on separate lines**."
        f"- **Never** nest 'Name:', 'Address:', or 'Description:' within a hyphenated list. These should always start a new line and be outside of any lists. "
        f"- Avoid combining these keywords with other sentences."
        f"Maintain consistent formatting throughout the entire itinerary. Use a single hyphen only for general list items, "
        f"such as describing activities or locations visited, but not for the **Name**, **Address**, or **Description** sections.")

    userPrompt = (
        f"Generate a travel itinerary for {userItinerary.duration} days, "
        f"starting from {userItinerary.origin} with the following destinations: {', '.join(userItinerary.destinations)}. "
        f"The total budget is {userItinerary.budget} dollars. The itinerary should not include the origin destination, but should only be"
        f"included as the description ")

    # Call the OpenAI API to generate the itinerary
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract itinerary content from response
    itineraryContent = response.choices[0].message.content
    print(f"content {itineraryContent}")

    # Find all names in response
    namePattern = r"\*\*Name:\*\*\s(.*?)(?=\n)"
    names: list[str] = re.findall(namePattern, itineraryContent)

    # Find all addresses in response
    addressPattern = r"\*\*Address:\*\*\s(.*?)(?=\n)"
    addresses: list[str] = re.findall(addressPattern, itineraryContent)

    # Get coordinates and googlePlaceId of addresses
    googlePlaceInfo = await geocodeLocations(addresses)

    # Places 
    places = []

    # Assign attributes to each place
    for id, (name, address, googlePlace) in enumerate(zip(names, addresses, googlePlaceInfo)):
        place = {
            "id": id,
            "type": "placeholder",
            "name": name,
            "address": address,
            "coordinates": googlePlace
        }

        places.append(place)

    # Prepare response data
    response_data = {
        "itinerary": itineraryContent,
        "places": places
    }

    print(response_data)
    # Return formatted response
    return response_data """

