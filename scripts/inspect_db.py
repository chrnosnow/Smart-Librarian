"""This script connects to a persistent ChromaDB instance, checks for a specific collection,
and retrieves information about the items stored in that collection."""
import chromadb
import pprint  # for nicely formatted dictionary printing

from smart_librarian.config import CHROMA_DB_PATH, COLLECTION_NAME


def main():
    # 1) Connect to the persistent ChromaDB client
    if not CHROMA_DB_PATH.exists():
        print(f"ChromaDB storage not found at: {CHROMA_DB_PATH}")
        print("Please run the main data embedding script first.")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # 2) Check if the collection exists
    print(f"Listing all collections in the database:")
    collections = client.list_collections()
    print([c.name for c in collections])

    if not any(c.name == COLLECTION_NAME for c in collections):
        print(f"\nCollection '{COLLECTION_NAME}' not found.")
        return

    # 3) Get the collection object
    collection = client.get_collection(name=COLLECTION_NAME)

    # 4) Get the total count of items
    count = collection.count()
    print(f"\nThe collection '{COLLECTION_NAME}' contains {count} items.")

    if count == 0:
        print("No collection found. Please run the embedding script first.")
        return

    # 5) Get the first few items using peek() for a quick sample
    print("\n--- Peeking at the first 5 items ---")
    peek_result = collection.peek(limit=5)
    pprint.pprint(peek_result)


"""
    # 6) Get ALL items in the collection
    # This can be memory-intensive for very large collections!
    # It returns a dictionary with all the ids, embeddings, metadatas, and documents.
    print("\n--- Getting ALL items from the collection ---")
    # We will only print the documents and metadatas to keep the output clean,
    # as printing thousands of embedding vectors would be too much.
    all_items = collection.get(
        include=["metadatas", "documents"]  # Specify what you want to retrieve
    )

    # Print details for the first 10 items for demonstration
    num_to_show = min(10, count)
    print(f"Showing details for the first {num_to_show} items retrieved:")
    for i in range(num_to_show):
        print(f"\nItem {i + 1}:")
        print(f"  ID: {all_items['ids'][i]}")
        print(f"  Metadata: {all_items['metadatas'][i]}")
        print(f"  Document: '{all_items['documents'][i][:100]}...'")  # Show first 100 chars
"""

if __name__ == "__main__":
    main()
