from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.smart_librarian.core.audio_service import generate_speech_stream  # Or from audio_service

router = APIRouter()


class TextToSpeechRequest(BaseModel):
    text: str


@router.post("/tts")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Endpoint to convert text to speech and stream the audio back.
    """
    try:
        stream = generate_speech_stream(request.text)
        # Use FastAPI's StreamingResponse to send the audio bytes
        # Set media_type to 'audio/mpeg' for MP3 files
        return StreamingResponse(stream.iter_bytes(), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error in TTS endpoint: {e}")
        return Response(content="Error generating speech", status_code=500)
