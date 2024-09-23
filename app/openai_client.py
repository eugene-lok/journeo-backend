from langchain.llms import OpenAI
from app.config import settings
import os

class OpenAIClient:
    def __init__(self):
        apikey = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=apikey)

    def get_location(self, prompt: str):
        response = self.client(prompt)
        return response
