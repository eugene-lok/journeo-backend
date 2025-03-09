# Journeo

---

## About
This is the backend for Journeo, an advanced LLM-based travel itinerary planner. It leverages OpenAI's GPT models, Google Places API, and Mapbox GL JS to generate personalized, intelligent travel itineraries with precise geolocation details.

### Key Features:
- **Intelligent Itinerary Generation**: Uses OpenAI's GPT to create personalized travel plans
- **Preference Extraction**: Advanced agent-based system to understand and extract user travel preferences
- **Geocoding Services**: Integrates Google's Geocoding and Places APIs to get precise location details
- **Route Calculation**: Uses Mapbox Directions API to calculate routes between destinations

Complementary frontend can be found in the [Journeo Frontend Repository](https://github.com/eugene-lok/journeo-frontend).

---

## Technologies Used 
- **Python**
- **FastAPI**
- **LangChain**
- **OpenAI GPT-4o**
- **Google Places API**
- **Mapbox Geocoding API**

---

## Setup / Installation 

**Note:** This is the backend component of Journeo. You must also set up the [Journeo Frontend](https://github.com/eugene-lok/journeo-frontend) for a complete application experience.

1. Prerequisites:
   - Python 3.8+
   - OpenAI Account
   - Google Cloud Platform Account
   - Mapbox Account

2. Obtain necessary API keys:
   - [Mapbox access token](https://docs.mapbox.com/help/getting-started/access-tokens/)
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Google Cloud API Key with Places API enabled](https://developers.google.com/maps/gmp-get-started)

3. Clone the repository:
    ```bash
    git clone https://github.com/eugene-lok/journeo-backend.git
    cd journeo-backend
    ```

4. Create a virtual environment:
    ```bash
    python3 -m venv env
    
    # Activate the environment
    # On Linux/Mac:
    source env/bin/activate
    
    # On Windows:
    \env\Scripts\activate
    ```

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Set up environment variables:
    - Create a `.env` file in the root of your project
    - Add the following keys:
    ```bash
    MAPBOX_ACCESS_TOKEN=your_mapbox_access_token
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

7. Run the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```

8. The API will be accessible on `http://localhost:8000`

## Frontend Repository
For the client-side implementation, visit the [Journeo Frontend Repository](https://github.com/eugene-lok/journeo-frontend).

## Key Components

### Preference Extraction Agent
- Uses LangGraph to create a state-based workflow for extracting travel preferences
- Handles complex, multi-turn conversations to gather trip details
- Ensures comprehensive information collection

### Itinerary Generation
- Leverages OpenAI's GPT-4o for intelligent itinerary creation
- Generates day-by-day travel plans with precise location details
- Calculates budget breakdowns and provides comprehensive travel guidance

### Geocoding and Route Services
- Uses Google Places API to get detailed location information
- Retrieves coordinates, place details, and photos
- Calculates routes between destinations using Mapbox Directions API
