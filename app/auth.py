from fastapi import APIRouter, HTTPException
from supabase import create_client
import os

# Initialize Supabase client
supabaseUrl = os.getenv("SUPABASE_URL")
supabaseKey = os.getenv("SUPABASE_KEY")
supabase = create_client(supabaseUrl, supabaseKey)

authRouter = APIRouter()

@authRouter.post("/api/signup")
async def signup(email: str, password: str):
    response = supabase.auth.sign_up({"email": email, "password": password})
    """ if response('error'):
        raise HTTPException(status_code=400, detail=response['error']['message']) """
    return {"message": "User signed up successfully"}

@authRouter.post("/api/login")
async def login(email: str, password: str):
    response = supabase.auth.sign_in_with_password({
    "email": email,
    "password": password
    })

    """ if response.error:
        raise HTTPException(status_code=400, detail=response.error.message)

    if not response.session or not response.session.access_token:
        raise HTTPException(status_code=500, detail="Failed to retrieve access token.")
 """
    return {
        "message": "Logged in successfully",
        "access_token": response.session.access_token
    }

@authRouter.post("/api/logout")
async def logout():
    supabase.auth.sign_out()
    return {"message": "Logged out successfully"}

