from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from smart_librarian.core.speech_service import transcribe_audio

router = APIRouter()


class TranscriptionResponse(BaseModel):
    """
    Defines the structure of the JSON response for audio transcription.
    """
    text: str


@router.post("/stt", response_model=TranscriptionResponse)
async def speech_to_text(audio_file: UploadFile = File(...)):
    """
    Endpoint to convert speech from an audio file to text.
    :param audio_file: audio_file (e.g., mp3, wav)
    :return: TranscriptionResponse containing the transcribed text.
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # Get the filename to check the extension
    filename = audio_file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="File has no name.")

    # Check if the extension is supported by OpenAI
    supported_extensions = ('.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm')
    if not filename.lower().endswith(supported_extensions):
        raise HTTPException(
            status_code=415,  # "Unsupported Media Type" HTTP status code
            detail=f"Unsupported file format. Please upload one of: {', '.join(supported_extensions)}"
        )

    try:
        transcribed_text = await transcribe_audio(audio_file)
        return TranscriptionResponse(text=transcribed_text)
    except Exception as e:
        print(f"Error in speech_to_text endpoint: {e}")
        error_detail = "Error transcribing audio."
        if "Unrecognized file format" in str(e):
            error_detail = "The audio file may be corrupted or in an unsupported format."
        raise HTTPException(status_code=500, detail=error_detail)
