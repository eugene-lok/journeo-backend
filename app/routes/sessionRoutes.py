from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

from app.models import SessionRequest
from app.loggerConfig import logger

from app.main import sessionManager, getOrCreateSession

router = APIRouter(prefix="/api", tags=["Sessions"])

@router.post("/validate-session/")
async def validateSession(sessionRequest: SessionRequest):
    try:
        sessionId = sessionRequest.sessionId
        if not sessionId:
            return JSONResponse(status_code=404, content={"valid": False})

        session = sessionManager.getSession(sessionId)
        if not session:
            return JSONResponse(status_code=404, content={"valid": False})

        # Check if session has expired
        currentTime = datetime.now()
        if (currentTime - session.lastAccessed) > timedelta(minutes=sessionManager.expirationMinutes):
            # Clean up expired session
            del sessionManager._sessions[sessionId]
            return JSONResponse(status_code=404, content={"valid": False})

        return JSONResponse(content={"valid": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-session/")
async def clearSession(sessionRequest: SessionRequest):
    try:
        sessionId = sessionRequest.sessionId
        if sessionId and sessionId in sessionManager._sessions:
            del sessionManager._sessions[sessionId]
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))