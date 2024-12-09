import os
import re
import httpx
import asyncio
import urllib.parse

# Geocode list of addresses in parallel requests and fetches place details in parallel
async def getAllPlaceDetails(names: list[str], addresses: list[str]):
    async with httpx.AsyncClient() as client:
        geocodeTasks = []
        autoCompleteTasks = []
        detailsTasks = []
        photoTasks = []
        results = []

        # Validate inputs
        if not names or not addresses or len(names) != len(addresses):
            print("Invalid input: names and addresses must be non-empty and of equal length")
            return []

        # Geocode addresses to obtain coordinates and placeId 
        for address in addresses:
            geocodeTask = getCoordinatesGoogle(client, address)
            geocodeTasks.append(geocodeTask)
        geocodeResults = await asyncio.gather(*geocodeTasks)
        
        # Filter out None results and get coordinates
        validGeoResults = [result for result in geocodeResults if result is not None]
        if not validGeoResults:
            print("No valid geocoding results found")
            return []
            
        # Use name and coordinates to fetch precise placeId
        for name, geocodeResult in zip(names[:len(validGeoResults)], validGeoResults):
            if not geocodeResult.get('coordinates'):
                autoCompleteTasks.append(None)
                continue
            autoCompleteTask = getPlaceFromAutocomplete(client, name, geocodeResult['coordinates'])
            autoCompleteTasks.append(autoCompleteTask)
        autoCompleteResults = await asyncio.gather(*autoCompleteTasks)

        # Fetch place details and photos
        for geocodeResult, autoCompleteResult in zip(validGeoResults, autoCompleteResults):
            try:
                # Initialize default result structure
                result = {
                    "initialPlaceId": geocodeResult.get('placeId'),
                    "coordinates": geocodeResult.get('coordinates'),
                    "placePrediction": None,
                    "photoUri": None,
                    "details": None
                }

                # Skip if no autocomplete result
                if autoCompleteResult is None:
                    results.append(result)
                    continue

                # Get place details
                detailsResult = await getPlaceDetailsFromId(client, autoCompleteResult['precisePlaceId'])
                
                # Update result with place details
                result['placePrediction'] = autoCompleteResult
                result['details'] = detailsResult

                # Get photo if available
                if detailsResult and detailsResult.get('photos'):
                    try:
                        photoName = detailsResult['photos'][0]['name']
                        photoResult = await getPlacePhotoFromPhotoName(client, photoName)
                        result['photoUri'] = photoResult
                    except Exception as e:
                        print(f"Error fetching photo: {str(e)}")
                        # Continue without photo if there's an error

                results.append(result)

            except Exception as e:
                print(f"Error processing place details: {str(e)}")
                # Add result with basic info if there's an error
                results.append({
                    "initialPlaceId": geocodeResult.get('placeId'),
                    "coordinates": geocodeResult.get('coordinates'),
                    "placePrediction": autoCompleteResult if autoCompleteResult else None,
                    "photoUri": None,
                    "details": None
                })

        # Check if airport for each place
        for place in results:
            checkIfAirport(place)

        return results


async def getCoordinatesGoogle(client, address):
    if not address:
        print("Empty address provided")
        return None
        
    try:
        googleAPIKey = os.getenv("GOOGLE_API_KEY")
        if not googleAPIKey:
            raise ValueError("Google API key not found")
            
        encoded_address = urllib.parse.quote(address)
        geocodeUrl = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={googleAPIKey}"
        
        response = await client.get(geocodeUrl)
        response.raise_for_status()  # Raises exception for 4XX/5XX status codes
        
        data = response.json()
        if not data.get("results"):
            print(f"No results found for address: {address}")
            return None
            
        location = data["results"][0]["geometry"]["location"]
        return {
            "coordinates": {
                "latitude": location["lat"],
                "longitude": location["lng"]
            },
            "placeId": data["results"][0]["place_id"]
        }
            
    except Exception as e:
        print(f"Error in getCoordinatesGoogle for address {address}: {str(e)}")
        return None

# Get precise place ID from Google Autocomplete API 
async def getPlaceFromAutocomplete(client, input, coordinates):
    if not input or not coordinates:
        return None
        
    try:
        apiKey = os.getenv("GOOGLE_API_KEY")
        if not apiKey:
            raise ValueError("Google API key not found")
            
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': apiKey,
            'X-Goog-FieldMask': "*"
        }
        
        # Try with location restriction first
        for radius in range(5000, 25000, 5000):
            body = {
                "input": input,
                "locationRestriction": {
                    "circle": {
                        "center": coordinates,
                        "radius": radius
                    }
                }
            }
            
            response = await client.post(
                "https://places.googleapis.com/v1/places:autocomplete",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('suggestions'):
                prediction = data['suggestions'][0]["placePrediction"]
                return {
                    "precisePlaceId": prediction["placeId"],
                    "text": prediction['text']['text']
                }
                
        # Fallback without location restriction
        response = await client.post(
            "https://places.googleapis.com/v1/places:autocomplete",
            headers=headers,
            json={"input": input}
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get('suggestions'):
            prediction = data['suggestions'][0]["placePrediction"]
            return {
                "precisePlaceId": prediction["placeId"],
                "text": prediction['text']['text']
            }
            
        return None
            
    except Exception as e:
        print(f"Error in getPlaceFromAutocomplete for input {input}: {str(e)}")
        return None
    
# Get place details from Google Place Details API using Place ID
async def getPlaceDetailsFromId(client, placeId):
    googleAPIKey = os.getenv("GOOGLE_API_KEY")
    fields = "id,displayName,primaryType,primaryTypeDisplayName,types,websiteUri,googleMapsUri,internationalPhoneNumber,nationalPhoneNumber,containingPlaces,regularOpeningHours,priceLevel,ratings,userRatingCount,photos"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': googleAPIKey,
        'X-Goog-FieldMask': fields
    }
    placeDetailsUrl = f"https://places.googleapis.com/v1/places/{placeId}"
    try:
        response = await client.get(placeDetailsUrl, headers=headers)
        if response.status_code == 200:
            result = response.json()
            #print(f"RESPONSE:{result}")
            if result is not None:
                ## TODO: Use new fields 
                return {
                    "id": result.get("id"),
                    "displayName": result.get("displayName"),
                    "primaryType": result.get("primaryType"),
                    "primaryTypeDisplayName": result.get("primaryTypeDisplayName"),
                    "types": result.get("types"),
                    "websiteUri": result.get("websiteUri"),
                    "googleMapsUri": result.get("googleMapsUri"),
                    "internationalPhoneNumber": result.get("internationalPhoneNumber"),
                    "nationalPhoneNumber": result.get("nationalPhoneNumber"),
                    "regularOpeningHours": result.get("regularOpeningHours"),
                    "priceLevel": result.get("priceLevel"),
                    "ratings": result.get("ratings"),
                    "userRatingCount": result.get("userRatingCount"),
                    "photos": result.get("photos"),
                }
            else:
                print(f"No result found for placeId: {placeId}")
                return None
        else:
            print(f"Error fetching place details for placeId {placeId}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching place details for placeId: {placeId}: {e}")
        return None

# Get place photo from Google Place Details API using Place ID and photo name
async def getPlacePhotoFromPhotoName(client, photoName):
    googleAPIKey = os.getenv("GOOGLE_API_KEY")
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': googleAPIKey,
    }
    placePhotoUrl = f"https://places.googleapis.com/v1/{photoName}/media?maxHeightPx=1600&skipHttpRedirect=true"
    try:
        response = await client.get(placePhotoUrl, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result is not None:
                return {
                    "photoDetails": result.get("photoUri")
                }
            else:
                print(f"No results found for photo name: {photoName}")
                return None
        else:
            print(f"Error fetching uri for photo {photoName}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching uri for photo {photoName}: {e}")
        return None

# Assigns isAirport attribute of place
def checkIfAirport(place):
    placeDetails = place["details"]
    if placeDetails is not None:
        primaryType = placeDetails.get("primaryType")
        types = placeDetails.get("types", [])
        if primaryType == "international_airport" or "international_airport" in types or "airport" in types:
            place["isAirport"] = True
        else:
            place["isAirport"] = False
    else:
        place["isAirport"] = False