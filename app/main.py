from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from functools import wraps

from app.models import UserItinerary, UserMessage
from app.middleware import addCorsMiddleware
from app.mapboxRoutes import getRouteFromMapbox
from app.loggerConfig import logger
from app.geoTools.geocoding import *
from app.extractorAgent import createTravelPreferenceWorkflow

import os
import httpx
import asyncio
import json
import uuid


# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

app = FastAPI()

# Apply CORS middleware
addCorsMiddleware(app)

class SessionData:
    def __init__(self):
        self.lastAccessed = datetime.now()
        self.createdAt = datetime.now()
        self.preferences = {}  # For travel preferences
        self.chatHistory = []  # For chat messages
        self.entities = {}     # For extracted entities

# Session manager class
class SessionManager:
    def __init__(self, expirationMinutes: int = 30):
        self._sessions: Dict[str, SessionData] = {}
        self.expirationMinutes = expirationMinutes

    def createSession(self) -> str:
        sessionId = str(uuid.uuid4())
        self._sessions[sessionId] = SessionData()
        return sessionId

    def getSession(self, sessionId: str) -> Optional[SessionData]:
        session = self._sessions.get(sessionId)
        if session:
            session.lastAccessed = datetime.now()
        return session

    def sessionExists(self, sessionId: str) -> bool:
        return sessionId in self._sessions

    def updateSession(self, sessionId: str, updateFunc) -> None:
        """
        Update session using a callback function
        """
        if session := self.getSession(sessionId):
            updateFunc(session)

    def cleanupExpiredSessions(self) -> None:
        currentTime = datetime.now()
        expired = [
            sessionId for sessionId, data in self._sessions.items()
            if (currentTime - data.lastAccessed) > timedelta(minutes=self.expirationMinutes)
        ]
        for sessionId in expired:
            del self._sessions[sessionId]

# Create global session manager
sessionManager = SessionManager()

# Base model for requests that include sessionId
class SessionRequest(BaseModel):
    sessionId: Optional[str] = None

# Dependency for session management
async def getOrCreateSession(sessionRequest: SessionRequest) -> tuple[str, SessionData]:
    """
    FastAPI dependency that either gets an existing session or creates a new one
    """
    sessionManager.cleanupExpiredSessions()
    
    sessionId = sessionRequest.sessionId
    if not sessionId or not sessionManager.sessionExists(sessionId):
        sessionId = sessionManager.createSession()
    
    session = sessionManager.getSession(sessionId)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create or retrieve session")
    
    return sessionId, session

# Updated request models
class UserMessage(SessionRequest):
    input: Union[str, Dict] 

class UserInputModel(SessionRequest):
    userInput: str

workflow = createTravelPreferenceWorkflow()   

@app.post("/api/extract-preferences/")
async def extractTravelPreferences(inputData: UserInputModel):
    try:
        sessionId, session = await getOrCreateSession(inputData)  # Pass the inputData directly
        
        config = {"configurable": {"thread_id": f"pref_{sessionId}"}}
        
        initialInput = {
            'userInput': inputData.userInput,
            'previousEntities': session.entities
        }
        
        result = workflow.invoke(initialInput, config=config)
        
        # Update session entities
        session.entities = result.get('extractedEntities', {})
        
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
    f"Every name you generate MUST be an exact, real name of a location. e.g. Don't name a location 'Departure from Location'.\n"
    f"Every address you generate MUST be an exact, real address. Don't use various places or a placeholder.\n"
    f"In your budgetBreakdown in your response, include a comprehensive description of how the budget is distributed throughout the trip."
)

systemMessage = SystemMessagePromptTemplate.from_template(systemPromptShort)
messageHistory = MessagesPlaceholder(variable_name="messages")
messagesList = []    

@app.post("/api/chat/")
async def chatResponse(message: UserMessage):
    try:
        print(f"Raw message received: {message}")
        sessionId, session = await getOrCreateSession(message)
        
        # Get stored entities from session
        stored_entities = session.entities
        print(f"Stored entities from session: {stored_entities}")

        # Prepare the chat prompt
        humanMessage = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages([
            systemMessage,
            humanMessage,
            messageHistory
        ])

        # Use stored entities if available, otherwise use input
        input_content = message.input
        if isinstance(input_content, dict):
            # Merge with stored entities to ensure we have complete data
            entities = stored_entities if stored_entities else input_content
            formatted_input = (
                f"Please create an itinerary for {entities.get('duration', 'N/A')} "
                f"in {entities.get('destinations', 'N/A')} "
                f"for {entities.get('numTravellers', '1')} traveler(s) "
                f"with a budget of {entities.get('budget', 'N/A')}. "
                f"Start date: {entities.get('startDate', 'N/A')}. "
                f"{'Includes children. ' if entities.get('includesChildren') == 'true' else ''}"
                f"{'Includes pets. ' if entities.get('includesPets') == 'true' else ''}"
            )
            input_content = formatted_input

        print(f"Formatted input for LLM: {input_content}")
        
        session.chatHistory.append(HumanMessage(content=str(input_content)))
        
        chain = prompt | llm
        response = chain.invoke({
            "input": input_content,
            "messages": session.chatHistory
        })


        if not response.content or not response.content.strip():
            raise HTTPException(status_code=500, detail="Empty or null JSON response.")

        try:
            itineraryContent = json.loads(response.content)
            
            # Validate itinerary structure
            for day in itineraryContent.get("days", []):
                if "day" not in day or "places" not in day or not isinstance(day["places"], list):
                    raise HTTPException(status_code=500, detail="Missing or invalid structure in 'days' element.")

            # Store response in session chat history
            session.chatHistory.append(AIMessage(content=response.content))
            
            # Process places and calculate routes
            places = await processItineraryPlaces(itineraryContent)
            routes = await calculateRoutes(places)

            # Ensure we return the same sessionId we received
            responseData = {
                "sessionId": sessionId,  # This will now match the input sessionId
                "itinerary": itineraryContent,
                "places": places,
                "routes": routes
            }

            return JSONResponse(content={"response": responseData})

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to decode JSON: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def processItineraryPlaces(itinerary_content: Dict) -> List[Dict]:
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

async def calculateRoutes(places: List[Dict]) -> List[Dict]:
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
                #print(f"results: {results}")
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

