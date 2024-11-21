from fastapi.middleware.cors import CORSMiddleware

def addCorsMiddleware(app):
    origins = [
        "https://journeo-frontend-zeta.vercel.app",  
        "http://localhost:3000",             # Local 
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
