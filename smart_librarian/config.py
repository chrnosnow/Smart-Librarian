from pathlib import Path

# ---------- Paths & Constants ---------- #

# Base directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to data directory
DATA_DIR = PROJECT_ROOT / "data"

# Path to the JSON file containing book summaries
JSON_PATH = DATA_DIR / "book_summaries.json"

# Path to the ChromaDB directory
CHROMA_DB_PATH = DATA_DIR / "chroma_storage"

# Path to the audio file for speech output
SPEECH_OUTPUT_PATH = DATA_DIR / "speech.mp3"

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

# Speech-to-Text model for transcription
STT_MODEL = "whisper-1"

# Image generation model
IMAGE_MODEL = "dall-e-3"
