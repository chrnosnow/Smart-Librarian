"""Simple CLI chatbot that uses RAG (Chroma) + Tool Calling to recommend a book and then fetch its full summary."""

import os
import platform
import asyncio
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

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
from smart_librarian.core.speech_service import transcribe_audio_sync
from smart_librarian.config import DATA_DIR

# --- Constants for Audio Recording ---
SAMPLE_RATE = 44100  # Hertz
RECORDING_PATH = DATA_DIR / "recording.wav"


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


def record_and_transcribe_audio() -> str:
    """Records audio from the microphone, saves it, and transcribes it."""
    try:
        # 1. Record Audio
        duration_str = input("Enter recording duration in seconds (e.g., 5): ")
        duration = int(duration_str)
        print("\nRecording...")
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # 2. Normalize audio ---
        # Convert to floating point numbers for calculations
        recording_float = recording.astype(np.float32)

        # Find the maximum absolute value in the recording
        max_amplitude = np.max(np.abs(recording_float))

        # If there's some sound (not total silence)
        if max_amplitude > 0:
            # Calculate the scaling factor to boost volume to 98% of max
            scaling_factor = 0.98 / max_amplitude
            # Apply the scaling factor
            normalized_recording_float = recording_float * scaling_factor
            # Convert back to int16 for saving as a WAV file
            normalized_recording = (normalized_recording_float * np.iinfo(np.int16).max).astype(np.int16)
        else:
            # If it's pure silence, do nothing
            normalized_recording = recording

        # 3. Save the normalized recording to a WAV file
        print(f"Saving recording to {RECORDING_PATH}...")
        write(RECORDING_PATH, SAMPLE_RATE, normalized_recording)

        # 3. Transcribe the saved file
        print("Transcribing audio...")
        transcribed_text = transcribe_audio_sync(RECORDING_PATH)
        print(f"Transcription successful: '{transcribed_text}'")
        return transcribed_text

    except ValueError:
        print("Invalid duration. Please enter a number.")
        return ""
    except Exception as e:
        print(f"An error occurred during voice input: {e}")
        return ""


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
        input_mode = input("Choose input mode: (1) Text or (2) Voice [type 'exit', 'quit' or 'q' to quit]: ").lower()

        if input_mode in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        user_query = ""
        if input_mode in ['1', 'text']:
            user_query = input("You: ")
        elif input_mode in ['2', 'voice']:
            user_query = record_and_transcribe_audio()
        else:
            print("Invalid input mode. Please choose '1' for text or '2' for voice.")
            continue

        # If transcription failed or user entered nothing, restart loop
        if not user_query:
            print("-" * 50)
            continue

        # 5. Profanity filter
        if not is_safe(text=user_query, client=rag_service.openai_client):
            print("Assistant: Please use appropriate language.")
            continue

        print("--- [Processing Request]... ---")
        bot_response = asyncio.run(rag_service.ask_book_chat(user_query, conversation_history))

        print(f"\nAssistant: {bot_response}\n")

        # 6. Text-to-Speech option
        listen_choice = input("Would you like to listen to the recommendation? (yes/no): ").lower()
        if listen_choice in ['yes', 'y']:
            play_audio_response(bot_response)

        print("-" * 50)


if __name__ == "__main__":
    main()
