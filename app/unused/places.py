import os
import asyncio
import httpx

# Get lat, long, and place_id from Google Geocoding API
async def getCoordinates(client, address):
    print(f"Address: {address}")
    api_key = os.getenv("GOOGLE_API_KEY")
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    try:
        response = await client.get(geocode_url)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                location = data["results"][0]["geometry"]["location"]
                place_id = data["results"][0]["place_id"]
                return {
                    "latitude": location["lat"],
                    "longitude": location["lng"],
                    "place_id": place_id,
                    "address": address  # Including address for reference
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

# Get place details from Google Text Search API
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
        "textQuery": textQuery
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

# Combine geocoding and place details retrieval for each address
async def geocodeAndGetDetails(client, address):
    coord = await getCoordinates(client, address)
    if coord and "place_id" in coord:
        details = await getPlaceDetails(client, address)
        if details:
            return details
        else:
            return coord  # Return coordinates if place details are unavailable
    else:
        return None  # Return None if geocoding fails

# Geocode list of addresses and get place details
async def geocodeLocations(addresses: list[str]):
    async with httpx.AsyncClient() as client:
        tasks = [geocodeAndGetDetails(client, address) for address in addresses]
        final_results = await asyncio.gather(*tasks)
    return final_results

# Example usage
if __name__ == "__main__":
    addresses = [
        "Capilano Suspension Bridge Park, 3735 Capilano Rd, North Vancouver, BC V7R 4J1, Canada",
        "Googleplex, 1600 Amphitheatre Parkway, Mountain View, CA",
        "Apple Park, 1 Infinite Loop, Cupertino, CA",
        "Nonexistent Place, Nowhere Land"
    ]
    results = asyncio.run(geocodeLocations(addresses))
    for result in results:
        if (result is not None):
            #print(result)
            continue

