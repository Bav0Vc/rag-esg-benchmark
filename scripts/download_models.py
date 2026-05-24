from sentence_transformers import SentenceTransformer

"""
Pre-downloads all embedding models into the HuggingFace cache.
Run once before the indexing pipeline:

  docker exec -it dynamic_rag_esg-app-1 python scripts/download_models.py
"""
MODELS = [
  "BAAI/bge-m3",
  "Snowflake/snowflake-arctic-embed-l-v2.0",
  "intfloat/multilingual-e5-large-instruct",
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Semantic chunking: document embedder
]

for model_id in MODELS:
  print(f"\nDownloading {model_id} ...")
  SentenceTransformer(model_id)
  print(f"✓ {model_id} cached")

print("\nAll models cached. Ready to run the indexing pipeline.")
