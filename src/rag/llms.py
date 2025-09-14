import os
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()

def get_openai_llm():
    """Returns OpenAI GPT model wrapper (requires OPENAI_API_KEY in env)."""
    return ChatOpenAI(
        model="gpt-4o-mini",  # or gpt-3.5-turbo, etc.
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

def get_local_llm(model_name="mistral"):
    """Returns a local LLM via Ollama."""
    return Ollama(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
