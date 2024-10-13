import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)