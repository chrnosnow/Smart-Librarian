"""Simple CLI chatbot that uses RAG (Chroma) + Tool Calling to recommend a book and then fetch its full summary."""

import json
import os
import platform
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from unidecode import unidecode

# Import settings from the central config file
from smart_librarian.config import (
    CHAT_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    JSON_PATH,
    OFFENSIVE_WORDS,
    OFFENSIVE_WORDS_NORMALIZED,
    SPEECH_OUTPUT_PATH,
    TTS_MODEL,
)

# --- Initial Loading and Client Setup ---

# Load environment variables (OPENAI_API_KEY) from .env file
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY not found in the .env file")

# Initialize the OpenAI client
openai_client = OpenAI()

# Connect to the existing ChromaDB persistent database
try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    print("Please run the 'scripts/embed_books.py' script first to create the database.")
    exit()

# Define variables for cache
embedding_cache = {}
response_cache = {}


# --- Data Loading and Tool Function Definitions ---

def load_full_summaries(path: Path) -> dict[str, str]:
    """Loads the full summaries from the JSON file into a dictionary."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Create a dictionary mapping each book's title to its summary
            return {item["title"]: item["summary"] for item in data}
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return {}


# Load the full summaries when the script starts
book_summaries_dict = load_full_summaries(JSON_PATH)


def get_summary_by_title(title: str) -> str:
    """
    (TOOL) Looks up a book by its exact title and returns the full summary.
    """
    print(f"\n--- [Tool Called]: get_summary_by_title(title='{title}') ---")
    summary = book_summaries_dict.get(title)
    if summary:
        return summary
    return f"The summary for the book '{title}' was not found."


def is_inappropriate(text: str) -> bool:
    """
    Checks if the user's input text contains any offensive words by comparing
    normalized (diacritic-free) versions of the words.
    """
    # Normalize the user's input text to be diacritic-free and lowercase
    normalized_text = unidecode(text.lower())

    # Find all whole words in the normalized input
    words_in_text = set(re.findall(r'\b\w+\b', normalized_text))

    # Check for any intersection between the words in the text and our offensive list
    # isdisjoint() is efficient for this task.
    return not words_in_text.isdisjoint(OFFENSIVE_WORDS_NORMALIZED)


# --- Core Chatbot Logic (RAG + Function Calling) ---

def ask_book_chat(query: str, messages: list) -> str:
    """
    Handles a single turn of conversation with the chatbot.
    Implements the RAG pipeline and Function Calling logic.
    """
    # 1. RAG - Retrieval: Find relevant documents in ChromaDB
    if query in embedding_cache:
        print("--- [Cache Hit]: Using cached embedding for the query. ---")
        query_embedding = embedding_cache[query]
    else:
        print("--- [Cache Miss]: Calling OpenAI API for new embedding. ---")
        query_embedding = openai_client.embeddings.create(
            input=[query], model=EMBEDDING_MODEL
        ).data[0].embedding
        embedding_cache[query] = query_embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=3)  # list with most relevant books
    retrieved_docs = results['documents'][0]

    if not retrieved_docs:
        return "I'm sorry, I couldn't find any books matching your interest. Could you please rephrase?"

    # 2. RAG - Augmentation: Construct the context for the prompt
    context = "\n\n---\n\n".join(retrieved_docs)  # concatenate the most relevant books to create the context for AI
    prompt = f"""
You are a friendly and helpful assistant specializing in book recommendations.
Your task is to recommend ONE book based on the user's interest, using ONLY the summaries provided below.

**First, provide a brief, 1-2 sentence explanation for your choice.**

After your explanation, you MUST call the `get_summary_by_title` tool to retrieve the detailed summary for the recommended book.

Available summaries (context):
---
{context}
---

User's interest: "{query}"
"""
    messages.append({"role": "user", "content": prompt})

    # 3. Generation: Call the LLM with the option to use tools
    # Define the tool for function calling
    # This tool will be used to fetch the detailed summary of the recommended book
    tools = [
        {
            "type": "function",
            # tool metadata that allows OpenAi to understand what this function does
            "function": {
                "name": "get_summary_by_title",
                "description": "Get the detailed summary for a specific book title.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The exact title of the book."
                        }
                    },
                    "required": ["title"],
                },
            },
        }
    ]

    # First API call
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, tools=tools, tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)  # Add the AI's response to the conversation history

    # 4. Function Calling: Check if the LLM wants to call our function
    if response_message.tool_calls:
        # extract function name and arguments from the tool call
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        if function_name == "get_summary_by_title":
            args = json.loads(tool_call.function.arguments)
            tool_response = get_summary_by_title(title=args.get("title"))

            # Second API call, providing the tool's result back to the LLM
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_response,
                }
            )
            final_response = openai_client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, max_tokens=250
            )
            return final_response.choices[0].message.content

    # Fallback in case no tool call was made
    return response_message.content or "I was unable to generate a response."


# --- Audio Features ---

def play_audio_response(text: str):
    """Generates audio for the given text and plays it."""
    print("--- [Generating Audio]... ---")
    try:
        #  create the streaming response for text-to-speech
        with openai_client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL, voice="alloy", input=text
        ) as response_stream:
            # stream the audio response to a file
            response_stream.stream_to_file(SPEECH_OUTPUT_PATH)

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
        if is_inappropriate(user_query):
            print("Assistant: Please use appropriate language.")
            continue

        # Get the bot's response
        if user_query in response_cache:
            print("--- [Cache Hit]: Using cached final response. ---")
            bot_response = response_cache[user_query]
        else:
            print("--- [Cache Miss]: Processing new request. ---")
            bot_response = ask_book_chat(user_query, conversation_history)
            response_cache[user_query] = bot_response  # Store the new response

        print(f"\nAssistant: {bot_response}\n")

        # 6. Text-to-Speech option
        listen_choice = input("Would you like to listen to the recommendation? (yes/no): ").lower()
        if listen_choice in ['yes', 'y']:
            play_audio_response(bot_response)

        print("-" * 50)


if __name__ == "__main__":
    main()
