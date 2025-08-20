from openai import OpenAI
from backend.smart_librarian.config import TTS_MODEL

# Initialize the OpenAI client
openai_client = OpenAI()


def generate_speech_stream(text: str):
    """
    Generates an audio stream from text using OpenAI TTS.
    Returns the streaming response object.
    """
    try:
        response_stream = openai_client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice="alloy",
            input=text
        )
        return response_stream

    except Exception as e:
        print(f"Error generating speech stream: {e}")
        raise  # Re-raise for the API endpoint to catch
