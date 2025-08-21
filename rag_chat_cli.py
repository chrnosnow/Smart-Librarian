"""Simple CLI chatbot that uses RAG (Chroma) + Tool Calling to recommend a book and then fetch its full summary."""

import os
import platform

from dotenv import load_dotenv

# --- Initial Loading and Client Setup ---

# Load environment variables (OPENAI_API_KEY) from .env file
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY not found in the .env file")

# Import services after loading environment variables
from smart_librarian.config import SPEECH_OUTPUT_PATH
from smart_librarian.core.rag_service import rag_service
from smart_librarian.core.audio_service import generate_speech_stream
from smart_librarian.core.helpers import is_safe


# --- Audio Features ---

def play_audio_response(text: str):
    """Generates audio for the given text and plays it."""
    print("--- [Generating Audio]... ---")
    try:
        #  create the streaming response for text-to-speech
        stream_generator = generate_speech_stream(text)
        # Stream the audio response to a file
        with open(SPEECH_OUTPUT_PATH, "wb") as f:
            for chunk in stream_generator:
                f.write(chunk)

        # OS-specific command to open the audio file
        system = platform.system()
        if system == "Windows":
            os.system(f"start {SPEECH_OUTPUT_PATH}")
        elif system == "Darwin":  # macOS
            os.system(f"open {SPEECH_OUTPUT_PATH}")
        else:  # Linux
            os.system(f"xdg-open {SPEECH_OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred while generating or playing the audio: {e}")


# --- Main Application Loop (CLI) ---

def main():
    """Runs the interactive command-line interface for the chatbot."""
    print(" Welcome to the Book Recommendation Assistant! ")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    # Maintain conversation history for better contextual responses
    conversation_history = [
        {"role": "system", "content": "You are a friendly assistant specializing in book recommendations."}
    ]

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # 5. Profanity filter
        if not is_safe(text=user_query, client=rag_service.openai_client):
            print("Assistant: Please use appropriate language.")
            continue

        print("--- [Processing Request]... ---")
        bot_response = rag_service.ask_book_chat(user_query, conversation_history)

        print(f"\nAssistant: {bot_response}\n")

        # 6. Text-to-Speech option
        listen_choice = input("Would you like to listen to the recommendation? (yes/no): ").lower()
        if listen_choice in ['yes', 'y']:
            play_audio_response(bot_response)

        print("-" * 50)


if __name__ == "__main__":
    main()
