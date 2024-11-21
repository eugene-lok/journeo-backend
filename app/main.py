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
    f"- **Number of travellers** (phrased as 'for X people, 'with X people', or 'X people going').\n"
    f"- **Trip budget** (phrased as '$X', 'a budget of X', or 'around X').\n\n"

    f"### Duration Handling:\n"
    f"**If the user specifies the duration in any form (e.g., 'X days', 'a week', or a range like 'from date A to date B'), assume the duration is complete and do not ask for it again.**\n\n"

    f"### Origin Handling:\n"
    f"**If the user specifies the origin with phrases like 'from X', 'starting in X', or 'origin is X', assume the origin is complete and do not ask for it again.**\n\n"

    f"### Destination Handling:\n"
    f"**If the user specifies the destination with phrases like 'to X', 'destination is X', or 'visit X', assume the destination is complete and do not ask for it again.**\n\n"

    f"### Number of Travellers Handling:\n"
    f"**If the user specifies the number of travellers with phrases like 'for X people, 'with X people', or 'X people going', assume the number of travellers is complete and do not ask for it again.**\n\n"

    f"### Budget Handling:\n"
    f"**If the user specifies the budget with phrases like '$X', 'a budget of X', or 'around X', assume the budget is complete and do not ask for it again.**\n\n"

    f"### Handling Changes:\n"
    f"- **If the user requests a change to their itinerary, do not reclarify all parameters unless explicitly asked to.**\n"
    f"- **First, ask the user: 'Would you like to keep the rest of the trip the same?'**\n"
    f"- If the user wants to keep the rest of the trip the same, make only the requested change and regenerate the itinerary.\n"
    f"- If the user wants to modify other aspects, clarify only the specific parameters they want to change and retain the rest of the inputs.\n\n"

    f"### Completion Handling:\n"
    f"Once the user provides all five parameters (trip duration, origin, destination, number of travellers, and budget), you must generate the itinerary immediately without asking further questions or clarifying anything. "
    f"Do not delay generating the itinerary once all the information has been gathered. Generate the itinerary only when all information has been gathered.\n\n"

    f"### Itinerary Format:\n"
    f"1. Title it 'Your Itinerary'. **This is mandatory. Do not use this phrase elsewhere.**\n"
    f"2. Organize the itinerary by days. The first and last days are for travel:\n"
    f"   - First day: Travel from the origin to the destination.\n"
    f"   - Last day: Travel back from the destination to the origin.\n"
    f"3. For each location you suggest, use the following mandatory format. Recommend at least 2 locations per day unless the single location will take a full day to visit:\n"
    f"   - **Name:** [Always start with this]\n"
    f"   - **Address:** [This must be on a new line]\n"
    f"   - **Description:** [Provide a brief description]\n\n"
    f"After the itinerary, include a 'Budget Breakdown' section.\n\n"
    f"Under no circumstances should you:\n"
    f"- Ask for additional information or preferences after all inputs are received.\n"
    f"- Delay generating the itinerary."
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

            # Get coordinates of addresses
            coordinates = await geocodeLocations(addresses)

            # Places
            places = []

            # Assign attributes to each place
            for id, (name, address, coordinate) in enumerate(zip(names, addresses, coordinates)):
                place = {
                    "id": id,
                    "name": name,
                    "address": address,
                    "coordinates": coordinate
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
    

# Generates itinerary and fetches coordinates
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

    # Get coordinates of addresses
    coordinates = await geocodeLocations(addresses)

    # Places 
    places = []

    # Assign attributes to each place
    for id, (name, address, coordinate) in enumerate(zip(names, addresses, coordinates)):
        place = {
            "id": id,
            "name": name,
            "address": address,
            "coordinates": coordinate
        }

        places.append(place)

    # Prepare response data
    response_data = {
        "itinerary": itineraryContent,
        "places": places
    }

    print(response_data)
    # Return formatted response
    return response_data

# Get lat, long from Mapbox Geocoding API 
async def getCoordinates(client, address):
    print(f"Address: {address}")
    accessToken = os.getenv("MAPBOX_ACCESS_TOKEN")
    geocodeUrl = f"https://api.mapbox.com/search/geocode/v6/forward?q={address}&access_token={accessToken}"
    response = await client.get(geocodeUrl)
    if response.status_code == 200:
        data = response.json()
        if data["features"]:
            coords = data["features"][0]["geometry"]["coordinates"]
            return {"latitude": coords[1], "longitude": coords[0]}
        return None

# Geocode list of addresses in parallel requests 
async def geocodeLocations(addresses: list[str]):
    async with httpx.AsyncClient() as client:
        tasks = [getCoordinates(client, address) for address in addresses]
        results = await asyncio.gather(*tasks)
    return results

# Get place types from Google Places API 

# Get GeoJSON routes from Mapbox Directions API
