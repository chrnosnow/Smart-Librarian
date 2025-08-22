from pathlib import Path

from openai import OpenAI, AsyncOpenAI
from fastapi import UploadFile

from smart_librarian.config import STT_MODEL

# Initialize the OpenAI client
# Client for synchronous functions (like the CLI helper)
sync_openai_client = OpenAI()

# Client for asynchronous functions (like the FastAPI endpoint)
async_openai_client = AsyncOpenAI()


# This function is designed to be used in an asynchronous context, such as a FastAPI endpoint
async def transcribe_audio(audio_file: UploadFile) -> str:
    """
    Transcribes an audio file using the OpenAI Whisper model.
    :param audio_file: the audio file uploaded by the user.
    :return: the transcribed text as a string.
    """
    try:
        # The async client needs the full file content in bytes
        file_content = await audio_file.read()
        file_tuple = (audio_file.filename, file_content)
        # Use the OpenAI client to transcribe the audio file
        transcription = await async_openai_client.audio.transcriptions.create(
            file=file_tuple,
            model=STT_MODEL
        )
        return transcription.text

    except Exception as e:
        print(f"Error during audio transcription: {e}")
        raise  # Re-raise for the API endpoint to catch


#  This function is designed to be used in a synchronous context, such as a script or a non-async function
def transcribe_audio_sync(file_path: Path) -> str:
    """
    Transcribes an audio file from a local path using the OpenAI Whisper model.
    :param file_path: the Path object pointing to the audio file.
    :return: the transcribed text as a string.
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcription = sync_openai_client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        print("Error during audio transcription:", e)
        raise
