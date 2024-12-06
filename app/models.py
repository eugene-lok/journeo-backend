from pydantic import BaseModel, Field
from typing import List, Optional

class UserItinerary(BaseModel):
    origin: str
    destinations: list[str]
    duration: int
    budget: int

class UserMessage(BaseModel):
    input: str

class Coordinates(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class Place(BaseModel):
    name: str
    address: str
    description: str
    coordinates: Optional[Coordinates] = None  # Default: None
    category: str
    duration: str
    cost_estimate: Optional[str] = None

class Day(BaseModel):
    day: int
    places: List[Place]

class RouteLeg(BaseModel):
    from_: Optional[Coordinates] = None
    to: Optional[Coordinates] = None
    distance: Optional[str] = None
    duration: Optional[str] = None
    transport_mode: Optional[str] = None

class Metadata(BaseModel):
    user_input: dict
    response_summary: Optional[str] = None

class ItineraryResponse(BaseModel):
    metadata: Metadata
    itinerary: List[Day]
    routes: Optional[List[RouteLeg]] = None  # Default: None