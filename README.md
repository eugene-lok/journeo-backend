# Journeo 

---

## About
This is the backend for Journeo, an LLM-based travel itinerary planner. It allows users to input their travel preferences such as budget, destinations, and duration to generate a personalized travel itinerary with OpenAI's GPT. Locations from this itinerary are displayed on an interactive map. 

### Key Features:
- Integrates OpenAI's API for dynamic itinerary generation based on user input.
- Utilizes Mapbox’s Geocoding API to asynchronously convert addresses to coordinates.
---

## Technologies Used 
- **Python** 
- **FastAPI**
- **Mapbox Geocoding API**
---

## Setup / Installation 

**Note:** This is only the backend of the application. You will need to set up and run the [frontend](https://github.com/eugene-lok/journeo-frontend) as well for full functionality.
1. Create a Mapbox account and obtain a [Mapbox access token](https://docs.mapbox.com/help/getting-started/access-tokens/).

2. Create an OpenAI account and obtain an [OpenAI API Key](https://platform.openai.com/api-keys).

3. Clone the repository:
    ```bash
    git clone https://github.com/eugene-lok/journeo-backend.git
    ```

4. Create a virtual environment in the root of the project:
    ```bash
    python3 -m venv env
    ```
    - *On Linux/Mac:*
    ```bash
    source env/bin/activate
    ```
    - *On Windows:*
    ```bash
    \env\Scripts\activate
    ```

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Set up environment variables:
    - Create a `.env` file in the root of your project with the following keys:
    ```bash
    MAPBOX_ACCESS_TOKEN=your_mapbox_access_token
    OPENAI_API_KEY=your_openai_api_key
    ```

7. Run the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```

8. The API will be accessible on `http://localhost:8000`. Ensure the frontend has been setup and is running as well. 

---
