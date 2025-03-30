import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from fastapi import HTTPException

from app.models import SessionRequest, SessionData

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

async def getOrCreateSession(sessionRequest: SessionRequest) -> Tuple[str, SessionData]:
    """FastAPI dependency that either gets an existing session or creates a new one"""
    sessionManager.cleanupExpiredSessions()
    
    sessionId = sessionRequest.sessionId
    if not sessionId or not sessionManager.sessionExists(sessionId):
        sessionId = sessionManager.createSession()
    
    session = sessionManager.getSession(sessionId)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create or retrieve session")
    
    return sessionId, session

# Create global instance
sessionManager = SessionManager()

