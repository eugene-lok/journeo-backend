from pydantic import BaseModel

class UserItinerary(BaseModel):
    origin: str
    destinations: list[str]
    duration: int
    budget: int

class UserMessage(BaseModel):
    input: str