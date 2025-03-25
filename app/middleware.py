import os
from fastapi.middleware.cors import CORSMiddleware

def addCorsMiddleware(app):
    origins = [
        os.getenv("PRODUCTION_ORIGIN"),  
        os.getenv("LOCAL_ORIGIN"),             # Local 
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
