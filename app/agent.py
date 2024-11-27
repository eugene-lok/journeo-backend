import os
from langchain_openai import ChatOpenAI
from langsmith import utils
from dotenv import load_dotenv

langchainApiKey = os.environ["LANGCHAIN_API_KEY"]
projectName = os.environ["LANGCHAIN_PROJECT"]

load_dotenv(dotenv_path=".env", override=True)
utils.tracing_is_enabled()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1500,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

llm.invoke("Hello, world!")