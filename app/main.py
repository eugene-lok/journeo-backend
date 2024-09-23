from openai import OpenAI
from fastapi import FastAPI, HTTPException
from app.models import UserItinerary
from app.middleware import addCorsMiddleware
import os
import re
import httpx
import asyncio

app = FastAPI()

# Apply CORS middleware
addCorsMiddleware(app)

# Generates itinerary and fetches coordinates
@app.post("/api/generateItinerary/")
async def generateItinerary(userItinerary: UserItinerary):
    # Prepare the prompt for the OpenAI API
    client = OpenAI()

    systemPrompt = (
        f"You are a helpful travel agent. You will be generating a highly specific itinerary based on parameters provided by the user. "
        f"The itinerary will be organized per day. Do not include any destinations in the origin, but dedicate the first and last days of "
        f"the itinerary to travel from origin to first the destination, and the final destination back to the origin, respectively. "
        f"For each location you recommend to visit, the following must be included: "
        f"- The name of the location"
        f"- An address for geocoding"
        f"- A short description of the location. "
        f"After the itinerary itself, include a section for a budget breakdown. For consistent formatting, your response will be parsed for the keywords "
        f"'Address: ' and 'Description: '."
        f"All keywords MUST be preceded by two asterisks, and followed by a colon, two asterisks, and a space. Example: **Address:** ")

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
            max_tokens=1000,
            temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(f"rawresponse {response}")

    # Extract itinerary content from response
    itineraryContent = response.choices[0].message.content
    print(f"content {itineraryContent}")
    # Find all addresses in response
    addressPattern = r"\*\*Address:\*\*\s(.*?)(?=\n)"
    addresses: list[str] = re.findall(addressPattern, itineraryContent)

    # Get coordinates of addresses
    coordinates = await geocodeLocations(addresses)

    # Prepare response data
    response_data = {
        "itinerary": itineraryContent,
        'addresses': addresses,
        "coordinates": coordinates
    }

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


