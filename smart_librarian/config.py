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

# --- Safety ---
# This list contains offensive, vulgar, and hateful terms in both English and Romanian
# for the purpose of content moderation.
# WARNING: The list below contains highly offensive language.
OFFENSIVE_WORDS = [
    # --- General Profanity & Vulgarity (English) ---
    "fuck",
    "fucking",
    "motherfucker",
    "shit",
    "bullshit",
    "piss",
    "cunt",
    "bitch",
    "asshole",
    "bastard",
    "damn",
    "hell",
    "wanker",
    "bollocks",
    "slut",
    "whore",
    "douche",
    "douchebag",

    # --- Racial & Ethnic Slurs (English) - HIGHLY OFFENSIVE ---
    # Note: These are some of the most harmful words.
    "nigger",
    "niggers",
    "nigga",
    "chink",
    "gook",
    "spic",
    "wetback",
    "kike",
    "faggot",
    "fag",
    "dyke",
    "tranny",
    "shemale",
    "retard",
    "retards",
    "retarded",
    "coon",
    "cracker",  # Can be considered a slur
    "jap",  # Derogatory when used as a slur

    # --- Profanity & Insults (Romanian) ---
    "prost",
    "proastă",
    "tâmpit",
    "tâmpită",
    "idiot",
    "idioată",
    "cretin",
    "cretină",
    "bou",
    "vacă",
    "cur",
    "pula",
    "pizda",
    "muie",
    "muiști",
    "coaie",
    "căcat",
    "rahat",
    "fut",
    "fute",
    "sugi",
    "sclav",
    "țigan",  # Offensive when used as a slur
    "cioară",  # Offensive when used as a slur
]

OFFENSIVE_WORDS_NORMALIZED = {unidecode(word) for word in OFFENSIVE_WORDS}
