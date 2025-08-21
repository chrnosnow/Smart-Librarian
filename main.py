import os

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables (OPENAI_API_KEY) from .env file
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY not found in the .env file")

from smart_librarian.api import audio, chat

app = FastAPI(title="Smart Librarian API")

app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(audio.router, prefix="/api", tags=["Audio"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Librarian API"}
