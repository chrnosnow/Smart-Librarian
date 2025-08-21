from openai import OpenAI
from smart_librarian.config import TTS_MODEL

# Initialize the OpenAI client
openai_client = OpenAI()


def generate_speech_stream(text: str):
    """
    Generates an audio stream from text using OpenAI TTS.
    Returns the streaming response object.
    """
    try:
        with openai_client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice="alloy",
                input=text
        ) as response_stream:
            # `yield from` passes the chunks from the stream directly to the caller
            yield from response_stream.iter_bytes()

    except Exception as e:
        print(f"Error generating speech stream: {e}")
        raise  # Re-raise for the API endpoint to catch
