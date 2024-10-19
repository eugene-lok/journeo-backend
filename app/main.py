from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.models import UserItinerary, UserMessage
from app.middleware import addCorsMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

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
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)
    
@app.post("/api/chat/")
async def chat_endpoint(message: UserMessage):

    systemPrompt = (
        f"You are a travel agent. Your job is to generate a complete itinerary based on the user’s input. "
        f"You must continue the conversation until the user provides all of the following information:\n"
        f"- Trip duration (days)\n"
        f"- Trip origin\n"
        f"- Trip destination (a single city or country)\n"
        f"- Trip budget\n\n"
        f"Once these inputs are received, you must generate the itinerary immediately without asking further questions. **Do not confirm or clarify anything.**"
        f"\n\nThe structure of the itinerary must follow this format:\n"
        f"1. Title it 'Your Itinerary'. **Do not use this phrase elsewhere.**\n"
        f"2. Organize the itinerary by days. The first and last days are for travel:\n"
        f"   - First day: Travel from the origin to the destination.\n"
        f"   - Last day: Travel back from the destination to the origin.\n"
        f"3. For each location you suggest, use the following mandatory format:\n"
        f"   - **Name:** [Always start with this]\n"
        f"   - **Address:** [This must be on a new line]\n"
        f"   - **Description:** [Provide a brief description]\n\n"
        f"After the itinerary, include a 'Budget Breakdown' section.\n\n"
        f"Under no circumstances should you:\n"
        f"- Ask for additional information or preferences after all inputs are received.\n"
        f"- Confirm or clarify information provided by the user.\n"
        f"- Delay generating the itinerary."
    )

    try:
        systemMessage = SystemMessagePromptTemplate.from_template(systemPrompt)
        humanMessage = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages([systemMessage, humanMessage])

        # Generate the response using the chain
        chain = prompt | llm
        response = chain.invoke({"input": message.input})

        itineraryContent = response.content

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
            response_data = {
                "itinerary": itineraryContent,
                "places": places
            }

            print(response_data)
            # Return formatted response
            return response_data

        # Return the raw response if it doesn't match the expected format
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


