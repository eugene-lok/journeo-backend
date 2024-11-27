import os
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from app.middleware import addCorsMiddleware
from app.models import UserMessage
import os
import re
import httpx
import asyncio

app = FastAPI()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

@app.post("/api/chat/")
async def chat_endpoint(message: UserMessage):

    systemPrompt = (
        f"You are a helpful travel agent. You will be generating a highly specific itinerary based a back-and-forth conversation with the user. "
        f"You must continue to converse with the user until they provide you the following trip info: "
        f"- Trip duration (days), "
        f"- Trip origin, "
        f"- Trip destination(s), "
        f"- Trip budget. "
        f"You will only generate an itinerary once this information has been provided. **ALWAYS** title the itinerary with 'Your Itinerary' ."
        f"**NEVER** use the phrase 'Your Itinerary' in any context otherwise."
        f"The itinerary you generate will be organized per day. Do not include any destinations in the origin, but dedicate the first and last days of "
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

    # Define the ChatPromptTemplate with both system and human prompts
    prompt = ChatPromptTemplate.from_messages (
        [("system", systemPrompt), ("human", message)]
    )

    #memory = ConversationBufferMemory(return_messages=True)

    try:
        # Prepare the input for the chain
        chain = prompt | llm
        response = chain.invoke({"input": message.input})  # Send user input to LangChain pipeline

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    print(response)
    return response
""" 
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
    return response_data """
    

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


