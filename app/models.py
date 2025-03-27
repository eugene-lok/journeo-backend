from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from datetime import datetime, timedelta

# Base model for requests that include sessionId
class SessionRequest(BaseModel):
    sessionId: Optional[str] = None

class UserMessage(SessionRequest):
    input: Union[str, Dict] 

# Eequest models
class UserInputModel(SessionRequest):
    userInput: str

class ChatRequest(BaseModel):
    sessionId: str
    entities: Dict[str, Any]  

class SessionData:
    def __init__(self):
        self.lastAccessed = datetime.now()
        self.createdAt = datetime.now()
        self.preferences = {}  # For travel preferences
        self.chatHistory = []  # For chat messages
        self.entities = {}     # For extracted entities

# Session manager class
class SessionManager:
    def __init__(self, expirationMinutes: int = 30):
        self._sessions: Dict[str, SessionData] = {}
        self.expirationMinutes = expirationMinutes

    def createSession(self) -> str:
        sessionId = str(uuid.uuid4())
        self._sessions[sessionId] = SessionData()
        return sessionId

    def getSession(self, sessionId: str) -> Optional[SessionData]:
        session = self._sessions.get(sessionId)
        if session:
            session.lastAccessed = datetime.now()
        return session

    def sessionExists(self, sessionId: str) -> bool:
        return sessionId in self._sessions

    def updateSession(self, sessionId: str, updateFunc) -> None:
        """
        Update session using a callback function
        """
        if session := self.getSession(sessionId):
            updateFunc(session)

    def cleanupExpiredSessions(self) -> None:
        currentTime = datetime.now()
        expired = [
            sessionId for sessionId, data in self._sessions.items()
            if (currentTime - data.lastAccessed) > timedelta(minutes=self.expirationMinutes)
        ]
        for sessionId in expired:
            del self._sessions[sessionId]

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