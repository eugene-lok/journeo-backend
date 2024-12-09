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

        # Geocode addresses to obtain coordinates and placeId 
        for address in addresses:
            geocodeTask = getCoordinatesGoogle(client, address)
            geocodeTasks.append(geocodeTask)
        geocodeResults = await asyncio.gather(*geocodeTasks)
        # Obtain coordinates from geocoded addresses
        coordinates = [geocodeResult['coordinates'] for geocodeResult in geocodeResults]
        
        # Use name and coordinates to fetch precise placeId
        for name, coordinate in zip(names, coordinates):
            autoCompleteTask = getPlaceFromAutocomplete(client, name, coordinate)
            autoCompleteTasks.append(autoCompleteTask)
        autoCompleteResults = await asyncio.gather(*autoCompleteTasks)

        # Obtain new place IDs from queried locations
        precisePlaceIds = []
        for autoCompleteResult in autoCompleteResults:
            if autoCompleteResult is not None:
                precisePlaceIds.append(autoCompleteResult['precisePlaceId'])
            else:
                precisePlaceIds.append(None)

        # Fetch place details using precisePlaceIds
        for precisePlaceId in precisePlaceIds:
            if precisePlaceId is not None:
                detailsTask = getPlaceDetailsFromId(client, precisePlaceId)
                detailsTasks.append(detailsTask)
            else:
                detailsTasks.append(None)
        # Await all details tasks
        detailsResults = []
        for detailsTask in detailsTasks:
            if detailsTask is not None:
                detailsResult = await detailsTask
                detailsResults.append(detailsResult)
            else:
                detailsResults.append(None)
        # Obtain place photo names from details
        placePhotoNames = []
        for detailsResult in detailsResults:
            if detailsResult is not None:
                placePhotoNames.append(detailsResult['photos'][0]['name'])
            else:
                placePhotoNames.append('N/A')
       
        # Fetch place photo uris from photo names
        for placePhotoName in placePhotoNames:
            if placePhotoName is not 'N/A':
                photoTask = getPlacePhotoFromPhotoName(client, placePhotoName)
                photoTasks.append(photoTask)
            else:
                photoTasks.append(None)
        # Await all photo tasks
        photoResults = []
        for photoTask in photoTasks:
            if photoTask is not None:
                photoResult = await photoTask
                photoResults.append(photoResult)
            else:
                photoResults.append(None)
        
        # Compile all results
        results = []
        for geocodeResult, autoCompleteResult, detailsResult, photoResult in zip(geocodeResults, autoCompleteResults, detailsResults, photoResults):
            result = {
                "initialPlaceId": geocodeResult['placeId'],
                "coordinates": geocodeResult['coordinates'],
                "placePrediction": autoCompleteResult,
                "photoUri": photoResult,
                "details": detailsResult
            }
            results.append(result)
    return results


async def getCoordinatesGoogle(client, address):
    print(f"Address: {address}")
    googleAPIKey = os.getenv("GOOGLE_API_KEY")
    
    # URL encode the address
    encoded_address = urllib.parse.quote(address)
    geocodeUrl = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={googleAPIKey}"
    
    try:
        response = await client.get(geocodeUrl)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                location = data["results"][0]["geometry"]["location"]
                placeId = data["results"][0]["place_id"]
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
    initialRadius = 5000
    maxRadius = 20000
    increment = 5000

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
 """
    radius = initialRadius
    while radius <= maxRadius:
        body = {
            "input": input,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": coordinates['latitude'],
                        "longitude": coordinates['longitude']
                    },
                "radius": radius
                }
            }
        }

        response = await client.post(autocompleteUrl, headers=headers, json=body)
        if response.status_code == 200 and response.content:
            try:
                data = response.json()
            except Exception as e:
                print(f"Exception occurred while parsing JSON response: {e}")
                radius += increment
                continue
            # Check if suggestions key exists
            suggestions = data.get('suggestions', [])
            if suggestions:
                prediction = suggestions[0]["placePrediction"]
                return {
                    "precisePlaceId": prediction["placeId"],
                    "text": prediction['text']['text']
                }
            else:
                print(f"No results found for query {input} with radius: {radius}m")
                radius += increment
        else:
            print(f"Error fetching results for query {input}: {response.text}")
            break

    # Attempt without location restriction 
    body = {
        "input": input,
    }

    response = await client.post(autocompleteUrl, headers=headers, json=body)
    print(f"FALLBACK: Attempting input {input} with no coordinates")
    if response.status_code == 200:
        try:
            data = response.json()
        except Exception as e:
            print(f"Exception occurred while parsing JSON response in fallback: {e}")
            return None
        # Check if suggestions key exists
        suggestions = data.get('suggestions', [])
        if suggestions:
            prediction = suggestions[0]["placePrediction"]
            return {
                "precisePlaceId": prediction["placeId"],
                "text": prediction['text']['text']
            }
        else:
            print(f"No results found for query {input} with no radius restriction.")
            return None
    else:
        print(f"Error fetching fallback results for query {input}: {response.text}")
        return None
    
# Get place details from Google Place Details API using Place ID
async def getPlaceDetailsFromId(client, placeId):
    googleAPIKey = os.getenv("GOOGLE_API_KEY")
    fields = "id,displayName,primaryType,primaryTypeDisplayName,types,websiteUri,googleMapsUri,internationalPhoneNumber,nationalPhoneNumber,containingPlaces,viewport,photos"
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
                    "viewport": result.get("viewport"),
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
                print(f"Photo Uri:{result.get("photoUri")}")
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