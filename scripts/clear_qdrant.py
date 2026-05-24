import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
collections = client.get_collections().collections

if not collections:
    print("No collections found.")
else:
    for col in collections:
        print(f"Deleting: {col.name}")
        client.delete_collection(col.name)
    print(f"\nDeleted {len(collections)} collection(s).")
