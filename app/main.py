from openai import OpenAI
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
from langchain.schema import AIMessage, HumanMessage, SystemMessage


#from app.auth import authRouter
import os
import re
import httpx
import asyncio

app = FastAPI()

# Apply CORS middleware
addCorsMiddleware(app)

# Add router
#app.include_router(authRouter)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

systemPrompt = (
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

    f"### Budget Handling:\n"
    f"**If the user specifies the budget with phrases like '$X', 'a budget of X', or 'around X', assume the budget is complete and do not ask for it again.**\n\n"

    f"### Handling Changes:\n"
    f"- **If the user requests a change to their itinerary, do not reclarify all parameters unless explicitly asked to.**\n"
    f"- **First, ask the user: 'Would you like to keep the rest of the trip the same?'**\n"
    f"  - **If the user responds affirmatively (e.g., 'Yes, keep the rest the same'), apply the requested change and regenerate the itinerary.**\n"
    f"  - **If the user wants to modify other aspects, ask specifically which parameters they would like to change and retain the rest of the inputs.**\n"
    f"- **Ensure that only the modified parameters are updated while others remain unchanged.**\n\n"

    f"### Completion Handling:\n"
    f"Once the user provides all five parameters (trip duration, origin, destination, number of travellers, and budget), you must generate the itinerary immediately without asking further questions or clarifying anything. "
    f"Do not delay generating the itinerary once all the information has been gathered. Generate the itinerary only when all information has been gathered.\n\n"

    f"### Itinerary Format:\n"
    f"1. Title it 'Your Itinerary'. **This is mandatory. Do not use this phrase elsewhere.**\n"
    f"2. Organize the itinerary by days. The first and last days are for travel:\n"
    f"   - First day: Travel from the origin to the destination.\n"
    f"   - Last day: Travel back from the destination to the origin.\n"
    f"3. For each location you suggest, use the following mandatory format. Recommend at least 2 locations per day unless the single location will take a full day to visit:\n"
    f"   - **Name:** [Always start with this.]\n"
    f"   - **Address:** [Provide the exact, real address on a new line. Do not use placeholders like '[Your Hotel Address]']\n"
    f"   - **Description:** [Provide a brief description]\n\n"
    f"4. **Ensure all addresses are precise and verifiable. For known locations like airports, use their official addresses.**\n\n"

    f"After the itinerary, include a 'Budget Breakdown' section.\n\n"

    f"Under no circumstances should you:\n"
    f"- Ask for additional information or preferences after all inputs are received.\n"
    f"- Delay generating the itinerary.\n"
    f"- Use placeholder text for addresses.\n"
    f"- Provide vague or imprecise addresses.\n"
)



systemMessage = SystemMessagePromptTemplate.from_template(systemPrompt)
messageHistory = MessagesPlaceholder(variable_name="messages")
messagesList = []
    
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

        itineraryContent = response.content

        # Store response in message list
        messagesList.append(AIMessage(content=response.content))

        # Check if the response contains the itinerary
        if "Your Itinerary" in itineraryContent:
            # Extract itinerary content from response
            print(f"content {itineraryContent}")

            # Find all names in response
            namePattern = r"\*\*Name:\*\*\s(.*?)(?=\n)"
            names: list[str] = re.findall(namePattern, itineraryContent)

            # Find all addresses in response
            addressPattern = r"\*\*Address:\*\*\s(.*?)(?=\n)"
            addresses: list[str] = re.findall(addressPattern, itineraryContent)

            # Get coordinates and googlePlaceId of addresses
            placesInfo = await getAllPlaceDetails(names, addresses)

            # Places
            places = []

            # Assign attributes to each place
            for id, (name, address, placeInfo) in enumerate(zip(names, addresses, placesInfo)):
                if 'airport' in name.lower():
                    placeType = "international_airport"
                else:
                    placeType = None
                place = {
                    "id": id,
                    "name": name,
                    "type": placeType,
                    "address": address,
                    "initialPlaceId": placeInfo["initialPlaceId"],
                    "coordinates": placeInfo['coordinates'],
                    "details": placeInfo['placePrediction']
                }
                places.append(place)

            # Prepare response data
            responseData = {
                "itinerary": itineraryContent,
                "places": places
            }

            print(responseData)
            # Return formatted response
            botResponse = {"response": responseData}
            return JSONResponse(content=botResponse)

        # Return raw response 
        botResponse = {"response": response.content}
        return JSONResponse(content=botResponse)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Geocode list of addresses in parallel requests and fetches place details in parallel
async def getAllPlaceDetails(names: list[str], addresses: list[str]):
    async with httpx.AsyncClient() as client:
        coordinates = []
        geocodeTasks = []
        autoCompleteTasks = []
        for name, address in zip(names, addresses):
            # Geocode addresses to obtain coordinates and placeId 
            geocodeTask = getCoordinatesGoogle(client, address)
            geocodeTasks.append(geocodeTask)
            # Get place details from combination of name of address
            #longAddress = f"{name}"
            # longAddress = f"{name}, {address}"
            #detailTasks = getPlaceDetails(client, longAddress)
            #autoCompleteTasks = getPlaceFromAutocomplete(client, longAddress)
            #allTasks.append(asyncio.gather(geocodeTasks,detailTasks))
            #allTasks.append(asyncio.gather(geocodeTasks, autoCompleteTasks))
        geocodeResults = await asyncio.gather(*geocodeTasks)
        coordinates = [geocodeResult['coordinates'] for geocodeResult in geocodeResults]
        
        for name, coordinate in zip(names, coordinates):
            # Use name and coordinates to fetch precise placeId
            autoCompleteTask = getPlaceFromAutocomplete(client, name, coordinate)
            autoCompleteTasks.append(autoCompleteTask)
        autoCompleteResults = await asyncio.gather(*autoCompleteTasks)

        # 
        #allResults = await asyncio.gather(*allTasks)
        results = []
        for geocodeResult, autoCompleteResult in zip(geocodeResults, autoCompleteResults):
            result = {
                "initialPlaceId": geocodeResult['placeId'],
                "coordinates": geocodeResult['coordinates'],
                "placePrediction": autoCompleteResult
            }
            results.append(result)

        print(f"geocoding results:\n{results}")
    return results

# Get lat, long, and place_id from Google Geocoding API
async def getCoordinatesGoogle(client, address):
    print(f"Address: {address}")
    googleAPIKey = os.getenv("GOOGLE_API_KEY")
    geocodeUrl = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={googleAPIKey}"
    try:
        response = await client.get(geocodeUrl)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                location = data["results"][0]["geometry"]["location"]
                placeId = data["results"][0]["place_id"]
                #print(data["results"][0]["geometry"])
                return {
                    "coordinates": {
                        "latitude": location["lat"],
                        "longitude": location["lng"]
                    },
                    "placeId": placeId,
                }
            else:
                print(f"No results found for address: {address}")
                return None
        else:
            print(f"Error fetching coordinates for address {address}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching coordinates for address {address}: {e}")
        return None

# Get precise place ID from Google Autocomplete API 
async def getPlaceFromAutocomplete(client, input, coordinates):
    print(f"\nname\n:{input}")
    apiKey = os.getenv("GOOGLE_API_KEY")
    autocompleteUrl = "https://places.googleapis.com/v1/places:autocomplete"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': apiKey,
        'X-Goog-FieldMask': "*"
    }
    
    """ body = {
        "input": input,
    }
 """# Restrict 2km within coordinates
    body = {
        "input": input,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": coordinates['latitude'],
                    "longitude": coordinates['longitude']
                },
            "radius": 2000.0
            }
        }
    }

    try:
        response = await client.post(autocompleteUrl, headers=headers, json=body)
        if response.status_code == 200:
            data = response.json()
            # Get long name and placeId from prediction
            if data["suggestions"]:
                prediction = data["suggestions"][0]["placePrediction"]
                return {
                    "precisePlaceId": prediction["placeId"],
                    "text": prediction['text']['text']
                }
            else:
                print(f"No results found for query: {input}")
                return None
        else:
            print(f"Error fetching results for query {input}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching places for query '{input}': {e}")
        return None
    

# Get place type and details from Google Text Search API
async def getPlaceDetails(client, textQuery):
    apiKey = os.getenv("GOOGLE_API_KEY")
    # Field mask
    fields = "places.id,places.displayName,places.formattedAddress,places.primaryType,places.googleMapsUri,places.websiteUri,places.rating,places.photos"
    # Construct request params and body
    textSearchUrl = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': apiKey,
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

# Get place details from Google Place Details API
""" async def getPlaceDetails(client, place_id):
    api_key = os.getenv("GOOGLE_API_KEY")
    fields = "place_id,name,types,rating,website,url,photos"
    place_details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields={fields}&key={api_key}"
    try:
        response = await client.get(place_details_url)
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                result = data["result"]
                return {
                    "id": result.get("place_id"),
                    "displayName": result.get("name"),
                    "types": result.get("types"),
                    "rating": result.get("rating"),
                    "websiteUri": result.get("website"),
                    "googleMapsUri": result.get("url"),
                    "photos": result.get("photos")  
                }
            else:
                print(f"No result found for placeId: {place_id}")
                return None
        else:
            print(f"Error fetching place details for placeId {place_id}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching place details for placeId: {place_id}: {e}")
        return None """