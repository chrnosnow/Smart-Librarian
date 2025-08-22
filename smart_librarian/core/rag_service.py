import json
from functools import lru_cache
from pathlib import Path

import chromadb
from openai import OpenAI

from smart_librarian.config import (
    CHAT_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    JSON_PATH
)


# --- The Service Class ---
# Encapsulates all logic and dependencies (clients, data, caches).
class RAGService:
    def __init__(self):
        """
        Initializes the service, setting up clients and loading data.
        This constructor is called once when the service instance is created.
        """
        print("Initializing RAGService...")
        # Initialize the OpenAI client
        self.openai_client = OpenAI()

        # Connect to the existing ChromaDB persistent database
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            # Re-raising the exception is often better than exit() in a library/service.
            raise ConnectionError("Failed to connect to ChromaDB.") from e

        # Load the full summaries when the script starts
        self.book_summaries_dict = self._load_full_summaries(JSON_PATH)
        print("RAGService initialized successfully.")

    def _load_full_summaries(self, path: Path) -> dict[str, str]:
        """Loads the full summaries from the JSON file into a dictionary."""
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Create a dictionary mapping each book's title to its summary
                return {item["title"]: item["summary"] for item in data}
        except FileNotFoundError:
            print(f"Error: The file at {path} was not found.")
            return {}

    def get_summary_by_title(self, title: str) -> str:
        """
        (TOOL) Looks up a book by its exact title and returns the full summary.
        """
        print(f"\n--- [Tool Called]: get_summary_by_title(title='{title}') ---")
        summary = self.book_summaries_dict.get(title)
        if summary:
            return summary
        return f"The summary for the book '{title}' was not found."

    # --- LRU Cache for Embeddings ---
    # The cache is now an instance method, wrapped with lru_cache.
    # `maxsize=128` means it will store the 128 most recent embeddings.
    @lru_cache(maxsize=128)
    def _get_embedding(self, query: str) -> list[float]:
        """
        Generates or retrieves a cached embedding for a given query.
        """
        print(f"--- [Embedding]: Processing query '{query}' ---")
        response = self.openai_client.embeddings.create(
            input=[query], model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    async def ask_book_chat(self, query: str, messages: list) -> str:
        """
        Handles a single turn of conversation with the chatbot.
        Implements the RAG pipeline and Function Calling logic.
        """
        # 1. RAG - Retrieval: Find relevant documents in ChromaDB
        query_embedding = self._get_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding],
                                        n_results=3)  # list with most relevant books
        retrieved_docs = results['documents'][0]
        if not retrieved_docs:
            return "I'm sorry, I couldn't find any books matching your interest. Could you please rephrase?"

        # 2. RAG - Augmentation: Construct the context for the prompt
        context = "\n\n---\n\n".join(retrieved_docs)  # concatenate the most relevant books to create the context for AI
        prompt = f"""
You are a specialized book recommendation assistant. Your task is to follow these rules strictly:

1.  Your ONLY source of information is the "Available Summaries" provided below. Do NOT use any of your own knowledge about books.
2.  You must analyze the user's interest and compare it against the provided summaries.
3.  **Decision Rule:**
    - **IF** you find a book summary that is a strong, direct match for the user's interest, you will recommend that ONE book.
    - **ELSE** (if there are no strong matches or the summaries are irrelevant to the user's interest), you MUST respond with: "I couldn't find a suitable book in my database for that specific interest. Could I help you with a different theme?" Do not try to recommend the 'closest' or 'most similar' book if it is not a good fit.

4.  **Format for a Successful Recommendation:**
    - First, provide a brief, 1-2 sentence explanation for why the book is a good match.
    - After your explanation, you MUST call the `get_summary_by_title` tool. Do not write the summary yourself.

---
Available summaries (context):
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
        response = self.openai_client.chat.completions.create(
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
                tool_response = self.get_summary_by_title(title=args.get("title"))

                # Second API call, providing the tool's result back to the LLM
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_response,
                    }
                )
                final_response = self.openai_client.chat.completions.create(
                    model=CHAT_MODEL, messages=messages, max_tokens=250
                )
                return final_response.choices[0].message.content

        # Fallback in case no tool call was made
        return response_message.content or "I was unable to generate a response."


# --- Singleton Instance ---
# Create a single instance of the service that the rest of the app can import and use.
# This ensures clients and data are loaded only once when the server starts.
rag_service = RAGService()
