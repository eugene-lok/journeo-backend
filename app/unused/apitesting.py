from openai import OpenAI
from fastapi import FastAPI
from models import LocationRequest, LocationResponse, UserItinerary
import os
import requests

def geocodeLocation(address):
    accessToken = os.getenv("MAPBOX_ACCESS_TOKEN")
    geocodeUrl = f"https://api.mapbox.com/search/geocode/v6/forward?q={address}&access_token={accessToken}"
    response = requests.get(geocodeUrl)
    if response.status_code == 200:
        data = response.json()
        print(data['features'][0]['geometry']['coordinates'])


example = "Kitayama, Fujinomiya, Shizuoka 418-0112"
geocodeLocation(example)

systemPrompt = ()

# userPrompt = (
#        f"Generate a travel itinerary for {itinerary.duration} days, "
#        f"starting from {itinerary.origin} with the following destinations: {', '.join(itinerary.destinations)}. "
#       f"The total budget is {itinerary.budget} dollars.")