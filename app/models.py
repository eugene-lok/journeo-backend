from pydantic import BaseModel

class UserItinerary(BaseModel):
    origin: str
    destinations: list[str]
    duration: int
    budget: int

class Coordinates(BaseModel):
    latitude: float
    longitude: float
class Place(BaseModel):
    id: int
    name: str
    address: str
    coordinates: list[Coordinates] 
    
class LoginRequest(BaseModel):
    email: str
    password: str
