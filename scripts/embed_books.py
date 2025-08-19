"""Embed book summaries and store them in ChromaDB."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# ---------- Load env vars early ---------- #
load_dotenv()  # os.getenv("OPENAI_API_KEY") can see .env

# Validate API key before making the client
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY not found – add it to .env or env vars.")

# ---------- OpenAI client ---------- #
openai_client = OpenAI()  # uses env var automatically
EMBEDDING_MODEL = "text-embedding-3-small"  # embedding model to use

# ---------- Paths & Constants ---------- #
BASE_DIR = Path(__file__).resolve().parent.parent  # root folder of the project
JSON_PATH = BASE_DIR / "data" / "book_summaries.json"  # path to the JSON file containing book summaries
CHROMA_DB_PATH = BASE_DIR / "chroma_storage"  # path to the ChromaDB directory
COLLECTION_NAME = "books"  # name of the collection in ChromaDB


# ---------- Helper functions ---------- #
def load_books(path: Path) -> List[Dict]:
    """Load JSON array of book objects from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts (single batch call)."""
    response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    # response.data is a list of objects with `.embedding` attribute
    return [item.embedding for item in response.data]


def add_to_collection_in_batches(collection, ids, embeddings, metadatas, documents, batch_size=100):
    """Add data to the collection in smaller batches to avoid memory issues."""
    for i in range(0, len(ids), batch_size):
        # This print statement is helpful to see the progress
        print(f"Adding batch {i // batch_size + 1}...")
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            documents=documents[i:i + batch_size]
        )


# ---------- Main ---------- #
def main():
    # 1) Load book objects
    books = load_books(JSON_PATH)
    if not books:
        raise ValueError("book_summaries.json is empty – nothing to embed.")

    # 2) Build embedding strings: include the title to enrich semantic signal
    texts = [f"{b['title']}. {b['summary']}" for b in books]

    # 3) Generate embeddings in batch
    embeddings = embed_texts(texts)

    # 4) Initialise (or connect to) local ChromaDB store
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # 5) Create / open the collection
    # Re-running the script won’t duplicate entries; Chroma handles that.
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 6) Prepare IDs (e.g. book_0) & minimal metadata (title)
    ids = [f"book_{i}" for i in range(len(books))]
    metadata = [{"title": b["title"]} for b in books]

    # 7) Insert / update documents + embeddings
    # documents saves the human text so we can return it later
    add_to_collection_in_batches(collection=collection, ids=ids, embeddings=embeddings, metadatas=metadata,
                                 documents=texts)

    # 8) Quick test
    query = "friendship and adventure"
    query_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    # 9) Pretty‑print results
    print(f"Top matches for: '{query}'\n")
    for doc_id, score, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
    ):
        print(f"{doc_id} | {meta['title']} | distance={score:.4f}")

    print(f"\nVectors stored in: {CHROMA_DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
