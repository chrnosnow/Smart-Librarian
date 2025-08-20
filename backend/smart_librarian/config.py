# config.py
from pathlib import Path
from unidecode import unidecode

# ---------- Paths & Constants ---------- #

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the JSON file containing book summaries
JSON_PATH = BASE_DIR / "data" / "book_summaries.json"

# Path to the ChromaDB directory
CHROMA_DB_PATH = BASE_DIR / "chroma_storage"

# Path to the audio file for speech output
SPEECH_OUTPUT_PATH = BASE_DIR / "speech.mp3"

# --- ChromaDB ---
# Name of the collection in ChromaDB
COLLECTION_NAME = "books"

# --- OpenAI Models ---
# Embedding model to use
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM model for the chatbot
CHAT_MODEL = "gpt-4.1-mini"

# Text-to-Speech model for audio output
TTS_MODEL = "tts-1"
