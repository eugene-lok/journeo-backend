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
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

app = FastAPI()

# Apply CORS middleware
addCorsMiddleware(app)

# Session storage 
class SessionData:
    def __init__(self, entities: Dict[str, Any]):
        self.entities = entities
        self.lastAccessed = datetime.now()
        self.createdAt = datetime.now()

class UserInputModel(BaseModel):
    userInput: str
    sessionId: Optional[str] = None

sessionStorage: Dict[str, SessionData] = {}

# Session cleanup function
def cleanupExpiredSessions(expirationMinutes: int = 30):
    currentTime = datetime.now()
    expired_sessions = [
        sessionId for sessionId, sessionData in sessionStorage.items()
        if (currentTime - sessionData.lastAccessed) > timedelta(minutes=expirationMinutes)
    ]
    for sessionId in expired_sessions:
        del sessionStorage[sessionId]

workflow = createTravelPreferenceWorkflow()   

@app.post("/api/extract-preferences/")
async def extract_travel_preferences(inputData: UserInputModel):
    try:
        # Clean up expired sessions first
        cleanupExpiredSessions()

        # Get or create session ID
        sessionId = inputData.sessionId
        if not sessionId or sessionId not in sessionStorage:
            sessionId = str(uuid.uuid4())
            sessionStorage[sessionId] = SessionData(entities={})
        
        # Update last accessed time
        sessionStorage[sessionId].lastAccessed = datetime.now()
        
        # Get previous entities for this specific session
        previousEntities = sessionStorage[sessionId].entities
        
        config = {"configurable": {"thread_id": f"pref_{sessionId}"}}
        
        initialInput = {
            'userInput': inputData.userInput,
            'previousEntities': previousEntities
        }
        
        result = workflow.invoke(initialInput, config=config)
        
        # Update session storage with new entities
        sessionStorage[sessionId].entities = result.get('extractedEntities', {})
        
        return {
            "sessionId": sessionId,
            "extractedEntities": result.get('extractedEntities', {}),
            "missingEntities": result.get('missingEntities', []),
            "isComplete": result.get('isComplete', False),
            "clarificationMessage": result.get('clarificationMessage', "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
                },
                "budgetBreakdown": {"type": "string"}
            },
            "required": ["days", "budgetBreakdown"],
            "additionalProperties": False
        },
        "strict": True,
    }
}

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=1500,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
    response_format=responseFormat
)

## Implement different budget options, don't have to ask for budget

systemPromptShort = (
    f"You are a travel agent. Your job is to generate a complete itinerary based on these parameters:"
    f"- **Trip duration** (phrased as 'X days', 'X-day trip', 'a week', or 'from date A to date B').\n"
    f"- **Trip origin** (phrased as 'from X', 'starting in X', or 'origin is X').\n"
    f"- **Trip destination** (phrased as 'to X', 'destination is X', or 'visit X').\n"
    f"- **Number of travellers** (phrased as 'for X people', 'with X people', 'X people going', 'alone', or 'solo').\n"
    f"- **Trip budget** (phrased as '$X', 'a budget of X', or 'around X').\n\n"
    f"Every address you generate MUST be an exact, real address. Don't use various places or a placeholder.\n"
    f"In your budgetBreakdown in your response, include a comprehensive description of how the budget is distributed throughout the trip."
)

systemMessage = SystemMessagePromptTemplate.from_template(systemPromptShort)
messageHistory = MessagesPlaceholder(variable_name="messages")
messagesList = []    


@app.post("/api/chat/")
async def chat_response(
    message: UserMessage,
    session_info: tuple[str, SessionData] = Depends(get_session)
):
    try:
        session_id, session = session_info
        
        # Prepare the chat prompt
        human_message = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages([
            systemMessage,
            human_message,
            messageHistory
        ])

        # Add human message and generate response
        session.messages.append(HumanMessage(content=message.input))
        chain = prompt | llm
        response = chain.invoke({
            "input": message.input,
            "messages": session.messages
        })

        # Validate response
        if not response.content or not response.content.strip():
            return None, "Error: Empty or null JSON response."

        try:
            itinerary_content = json.loads(response.content)
            print(f"\nResponse Formatted: {response.content}")
            
            # Validate itinerary structure
            for day in itinerary_content.get("days"):
                if "day" not in day or "places" not in day or not isinstance(day["places"], list):
                    return None, "Error: Missing or invalid structure in 'days' element."

            # Store response in message list
            session.messages.append(AIMessage(content=response.content))
            
            # Process places and calculate routes
            places = await process_itinerary_places(itinerary_content)
            routes = await calculate_routes(places)

            # Prepare response data
            response_data = {
                "sessionId": session_id,
                "itinerary": itinerary_content,
                "places": places,
                "routes": routes
            }

            return JSONResponse(content={"response": response_data})

        except json.JSONDecodeError as e:
            return None, f"Error: Failed to decode JSON. {str(e)}"
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def process_itinerary_places(itinerary_content: Dict) -> List[Dict]:
    """Extract and process place information from itinerary content."""
    names = []
    addresses = []
    for day in itinerary_content["days"]:
        for place in day["places"]:
            names.append(place["name"])
            addresses.append(place["address"])

    # Get coordinates and googlePlaceId of addresses
    places_info = await getAllPlaceDetails(names, addresses)
    
    places = []
    # Assign attributes to each place
    for id, (name, address, place_info) in enumerate(zip(names, addresses, places_info)):
        place = {
            "id": id,
            "name": name,
            "isAirport": None,
            "address": address,
            "initialPlaceId": place_info['initialPlaceId'],
            "coordinates": place_info['coordinates'],
            "predictedLocation": place_info['placePrediction'],
            "details": place_info['details']
        }
        checkIfAirport(place)
        places.append(place)
    
    return places

async def calculate_routes(places: List[Dict]) -> List[Dict]:
    """Calculate routes between consecutive places, excluding airport-to-airport routes."""
    routes = []
    
    if len(places) < 2:
        return routes

    # Create pairs of consecutive places
    consecutive_pairs = [(places[i], places[i + 1]) for i in range(len(places) - 1)]
    
    # Filter out pairs where both are airports
    non_airport_pairs = [
        (from_place, to_place) for from_place, to_place in consecutive_pairs
        if not (from_place.get("isAirport", False) and to_place.get("isAirport", False))
    ]

    async with httpx.AsyncClient() as client:
        route_tasks = [
            getRouteFromMapbox(
                client,
                startCoords=from_place["coordinates"],
                endCoords=to_place["coordinates"]
            )
            for from_place, to_place in non_airport_pairs
        ]
        
        route_results = await asyncio.gather(*route_tasks, return_exceptions=True)

        # Process route results
        for (from_place, to_place), result in zip(non_airport_pairs, route_results):
            if isinstance(result, Exception):
                print(f"Failed to fetch route between {from_place['name']} and {to_place['name']}: {result}")
                continue
            
            if result:
                routes.append({
                    "from": {
                        "id": from_place["id"],
                        "name": from_place["name"],
                        "coordinates": from_place["coordinates"]
                    },
                    "to": {
                        "id": to_place["id"],
                        "name": to_place["name"],
                        "coordinates": to_place["coordinates"]
                    },
                    "route": result
                })
            else:
                print(f"No route found between {from_place['name']} and {to_place['name']}.")

    return routes


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

