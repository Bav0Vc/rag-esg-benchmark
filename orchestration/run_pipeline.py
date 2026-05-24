"""
End-to-end pipeline orchestration: clears Qdrant, indexes documents, runs the query benchmark,
and scores results with RAGAS. Intended for a clean full run; for partial or resumed runs,
call the individual stage modules directly.

    python -m orchestration.run_pipeline
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from scripts.logger import setup_logging
from evaluation.ragas_eval import evaluate_results
from pipeline.indexing_pipeline import run_indexing
from orchestration.benchmark_loop import run_benchmark

load_dotenv()
setup_logging("run_pipeline")


def clear_qdrant_collections():
  """Wipes all collections from the Qdrant database."""
  try:
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    collections = client.get_collections().collections
    if not collections:
      print("No collections found in Qdrant. Nothing to clear.")
      return

    print(f"Found {len(collections)} collections. Deleting all...")
    for collection in collections:
      client.delete_collection(collection.name)
      print(f"  Deleted collection: {collection.name}")
    print("All collections cleared from Qdrant.")

  except Exception as exc:
    print(f"  [error] An error occurred while clearing Qdrant collections: {exc}")
    print("  Please check your Qdrant connection details and ensure the server is running.")
    sys.exit(1)


print("Starting automated pipeline:")

# ── Step 1: Clear existing vector database collections ───────────────────────
print("\n" + "=" * 60)
print("STEP 1: Clearing Qdrant vector database")
print("=" * 60)
clear_qdrant_collections()

# ── Step 2: Index all chunker × embedder combinations ────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Indexing pipeline")
print("=" * 60)
run_indexing(resume_from=0)

# ── Step 3: Run query benchmark over all configurations ──────────────────────
print("\n" + "=" * 60)
print("STEP 3: Benchmark loop")
print("=" * 60)
run_benchmark()

# ── Step 4: RAGAS evaluation ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: RAGAS evaluation")
print("=" * 60)
asyncio.run(evaluate_results())


print("\nPipeline finished successfully.")
sys.exit()
