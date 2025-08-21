from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# Import the core RAG logic from your service layer
# This is a key principle of clean architecture: the API layer depends on the service layer.
from smart_librarian.core.rag_service import rag_service
from smart_librarian.core.helpers import is_safe

router = APIRouter()


# --- Pydantic Models ---
# These models define the "shape" of the JSON data for Smart Librarian API.
# FastAPI uses them for automatic validation and documentation.

class ChatRequest(BaseModel):
    """Defines the structure of an incoming chat request from the frontend."""
    query: str
    # Optional: add a conversation_history list here later if needed
    # history: list = [] 


class ChatResponse(BaseModel):
    """Defines the structure of the JSON response sent back to the frontend."""
    answer: str
    # add more fields here later for other features, e.g., imageUrl
    # imageUrl: str | None = None


# --- API Endpoint ---

@router.post("/chat", response_model=ChatResponse)
async def handle_chat_request(request: ChatRequest):
    """
    Handles an incoming chat query from the user.
    
    This endpoint performs the following steps:
    1. Validates the incoming request data against the ChatRequest model.
    2. Checks the user's query for inappropriate content.
    3. Calls the core RAG service to get a book recommendation.
    4. Returns the response in the format defined by ChatResponse.
    """
    user_query = request.query

    # 1. Input validation and safety check
    if not user_query or user_query.isspace():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not is_safe(text=user_query, client=rag_service.openai_client):
        return ChatResponse(answer="I can only discuss topics related to books. Please use appropriate language.")

    try:
        # 2. Call the core RAG service to get the answer
        # We create a new, empty conversation history for each request for a stateless API.
        # For a stateful chat, you would manage this history differently.
        conversation_history = [
            {"role": "system", "content": "You are a friendly assistant specializing in book recommendations."}
        ]

        # Call the RAG service to get the book recommendation
        response_text = rag_service.ask_book_chat(user_query, conversation_history)

        # 3. Return the structured response
        return ChatResponse(answer=response_text)

    except Exception as e:
        # If anything goes wrong in the RAG service, we catch it here
        # and return a generic server error to the frontend.
        # It's good practice to log the actual error `e` on the server for debugging.
        print(f"An unexpected error occurred in the chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")
