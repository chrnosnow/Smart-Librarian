from fastapi import FastAPI
from backend.smart_librarian.api import chat
from backend.smart_librarian.api import audio

app = FastAPI(title="Smart Librarian API")

app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(audio.router, prefix="/api", tags=["Audio"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Librarian API"}
