import os
import httpx

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

async def getRouteFromMapbox(client: httpx.AsyncClient, startCoords, endCoords):

    mapboxUrl = "https://api.mapbox.com/directions/v5/mapbox/driving"
    start = f"{startCoords['longitude']},{startCoords['latitude']}"
    end = f"{endCoords['longitude']},{endCoords['latitude']}"
    url = f"{mapboxUrl}/{start};{end}"
    params = {
        "geometries": "geojson",
        "access_token": MAPBOX_ACCESS_TOKEN
    }

    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            # Return first route only
            return data["routes"][0]["geometry"]
        else:
            print(f"No routes found between {start} and {end}.")
            return None
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP error occurred while fetching route: {http_err}")
    except Exception as err:
        print(f"An error occurred while fetching route: {err}")
    return None
