import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables (OPENAI_API_KEY) from .env file
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY not found in the .env file")

from smart_librarian.api import audio, chat, speech
from smart_librarian.config import SPEECH_OUTPUT_PATH


# --- Lifespan Manager ---
# This is the modern way to handle startup/shutdown events in FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup complete.")

    yield  # The application runs while the code is yielded

    print("Application shutting down. Cleaning up...")
    if SPEECH_OUTPUT_PATH.exists():
        try:
            os.remove(SPEECH_OUTPUT_PATH)
            print(f"Successfully deleted temporary file: {SPEECH_OUTPUT_PATH}")
        except OSError as e:
            print(f"Error deleting file {SPEECH_OUTPUT_PATH}: {e}")


# --- Create the App with the Lifespan ---
app = FastAPI(title="Smart Librarian API", lifespan=lifespan)

# --- Include Routers ---
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(audio.router, prefix="/api", tags=["Audio"])
app.include_router(speech.router, prefix="/api", tags=["Speech-to-Text"])


# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Librarian API"}
