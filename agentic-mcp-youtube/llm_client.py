import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

PROFILE = os.getenv("PROFILE", "local")  # default to local if not set

if PROFILE == "openai":
    API_KEY = os.getenv("OPENAI_API_KEY_OPENAI")
    API_BASE = os.getenv("OPENAI_API_BASE_OPENAI")
    MODEL = os.getenv("OPENAI_MODEL_OPENAI")
elif PROFILE == "local":
    API_KEY = os.getenv("OPENAI_API_KEY_LOCAL")
    API_BASE = os.getenv("OPENAI_API_BASE_LOCAL")
    MODEL = os.getenv("OPENAI_MODEL_LOCAL")
else:
    raise ValueError(f"Unknown PROFILE={PROFILE}, must be 'openai' or 'local'")

# Create OpenAI-compatible client (works for both)
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def chat_completion(messages, temperature=0.0, max_tokens=500):
    """
    Wrapper for chat completions that works across OpenAI and LM Studio.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content