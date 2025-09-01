# Smart Librarian

An intelligent book recommendation chatbot powered by RAG, Function Calling, and Multimodal AI.
It leverages a curated database of book summaries to provide personalized recommendations, detailed summaries, and
evocative cover images. The system supports both text and voice interactions, ensuring an engaging user experience.

## Key Features

This application fulfills a comprehensive set of requirements, showcasing modern AI and web development practices:

- **RAG-Powered Recommendations**: Utilizes a Retrieval-Augmented Generation (RAG) pipeline with a **ChromaDB** vector
  store. The chatbot provides recommendations based *only* on a curated database of book summaries, ensuring grounded
  and relevant answers.

- **Detailed Summaries with Tool Calling**: After recommending a book, the AI model uses **OpenAI Function Calling**
  to execute a Python tool (`get_summary_by_title`) that retrieves the full, detailed summary for the recommended book.

- **Content Moderation**: All user inputs are checked for inappropriate language. The backend includes a safety
  filter using OpenAI's Moderation API to ensure conversations remain on-topic and respectful.

- **Text-to-Speech (TTS)**: Users can listen to the chatbot's text responses. The backend provides an endpoint that
  converts text into high-quality streaming audio using OpenAI's TTS model.

- **Speech-to-Text (STT)**: Users can interact with the chatbot using their voice. The application supports audio
  uploads, which are transcribed into text using the **OpenAI Whisper** model.

- **AI Image Generation**: To enrich the user experience, the system automatically generates a unique, evocative book
  cover image for each recommendation using **DALL-E 3**.

- **Robust Backend API**: The entire application is powered by a backend built with **FastAPI**, exposing clean,
  well-documented, and asynchronous API endpoints for all functionalities.

- **Interactive CLI Client**: A fully-featured command-line interface (`rag_chat_cli.py`) is available for testing and
  demonstrating all backend features, including voice input/output.

- **Modern Frontend**: The architecture is designed to support a decoupled frontend. All backend
  functionalities are exposed via API, ready to be consumed by a web interface.

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI & ML**:
    - OpenAI API (GPT models for chat, DALL-E 3 for images, Whisper for STT, TTS)
    - Retrieval-Augmented Generation (RAG)
- **Database**: ChromaDB (Persistent Vector Store)
- **CLI Tools**: `sounddevice`, `scipy` for audio recording
- **Frontend**: React + Vite, JavaScript
- **Containerization**: Docker, Docker Compose

## Project Structure

The project follows a clean, scalable structure:

- `smart_librarian/`: The core Python package.
    - `api/`: Contains FastAPI routers for each feature (chat, audio, stt).
    - `core/`: Encapsulates all business logic in services (RAG, audio, etc.).
    - `config.py`: Centralized configuration for models, paths, and constants.
- `scripts/`: Utility scripts for tasks like data embedding (`embed_books.py`).
- `data/`: Stores persistent data like the source JSON, ChromaDB files, and temporary audio files.
- `main.py`: The entry point for the FastAPI application.
- `rag_chat_cli.py`: The standalone command-line client.
- `Dockerfile`: Defines the image for the backend service.
- `frontend/Dockerfile`: Defines the image for the frontend service.
- `docker-compose.yml`: Orchestrates the multi-container application.

## Getting Started

### Prerequisites

- Docker and Docker Compose (or a compatible tool like Rancher Desktop). This is the recommended way to run the project.
- Python 3.9+
- Node.js and npm
- An OpenAI API key

### Installation & Setup

#### I. Docker (Recommended)

*This method runs the entire application in containers, so you do not need to install Python or Node.js on your
machine.*

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/smart-librarian.git
   cd smart-librarian
   ```
2. **Set up your environment variables:**
    - Create a file named `.env` in the project root.
    - Copy the contents of `.env.example` into `.env`.
    - Add your OpenAI API key to the `.env` file:
      ```env
      OPENAI_API_KEY="sk-..."
      ```

### Usage

1. **Embed the Book Data (Run this first!)**
   Before starting the application, you must populate the vector database with the book summaries. We will run the
   script inside a temporary container.
   ```bash
   docker-compose run --rm api python scripts/embed_books.py
   ```

2. **Build and run the Full-Stack Application**
   This will build the images and start the backend and frontend services.
   ```bash
   docker-compose up --build
   ```
    - The Backend API will be available at http://localhost:8000.
    - The Frontend App will be available at http://localhost:3000.

#### II. Local Setup (Without Docker)

Follow these steps if you prefer to run the backend natively.

1. **Follow steps 1 and 2** from the Docker setup.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Backend Server**
   This will start the FastAPI server on `http://localhost:8000`.
   ```bash
   uvicorn main:app --reload
   ```
   You can access the interactive API documentation at `http://localhost:8000/docs`.

5. **Run the CLI Client (Optional)**
   To interact with the chatbot directly from your terminal:
   ```bash
   python rag_chat_cli.py
   ```

## API Endpoints

The backend exposes the following main endpoints:

| Method | Path        | Description                                          |
|--------|-------------|------------------------------------------------------|
| `POST` | `/api/chat` | Sends a text query, gets a recommendation and image. |
| `POST` | `/api/tts`  | Converts text to streaming audio.                    |
| `POST` | `/api/stt`  | Transcribes an uploaded audio file to text.          |

## Example Usage

# Successful chat response:

When a user's query matches a book in the database, the chatbot provides a concise recommendation, a full summary, and a
generated cover image.

**Normal chat**
![alt text="A screenshot of the Smart Librarian chat interface. The chat shows a user query and a successful book recommendation from the bot, including a summary and a generated image."](prompt_examples/successful_result_ui.png)

**Accessibility mode chat**
![alt text="A screenshot of the Smart Librarian chat interface in high-contrast accessibility mode. The chat shows a user query and a successful book recommendation from the bot, including a summary and a generated image."](prompt_examples/successful_result_accessibility_mode_ui.png)

# Book not found:

When a user's query does not match any book in the database, the chatbot responds appropriately, indicating that no
recommendation can be made.

![alt text="Screenshot of the Smart Librarian chat, showing the bot's response when no suitable book is found in the database."](prompt_examples/book_not_found_ui.png)

## To-Do & Future Work

-   [ ] **Stream TTS to Browser**: Implement direct audio streaming to the browser instead of just providing a link.
-   [ ] **Real-time Microphone Streaming**: Enhance the STT feature on the frontend to use the browser's microphone in
    real-time.
-   [ ] **User Authentication**: Add a login system to manage conversation history for individual users.
