"""
Indexing pipeline: converts raw ESG documents via the Unstructured API, chunks, embeds, and writes
to Qdrant. Covers 15 unique chunker × embedder combinations (6 non-semantic + 9 semantic).

SemanticEmbeddingChunker runs in two separate Haystack pipelines — chunking first, then embedding —
because both models need GPU and loading them simultaneously causes OOM on 8 GiB cards.
"""
import gc
import os
from pathlib import Path
from haystack import Pipeline
from itertools import product
from dotenv import load_dotenv
from haystack.utils import Secret
from scripts.logger import setup_logging
# Converter & Cleaner
from pipeline.components.document_cleaner import DocumentCleaner, ChunkMetaCleaner
from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter
# Chunking
from pipeline.components.chunking.fixed import FixedSizeTokenSplitter
from pipeline.components.chunking.semantic import SemanticEmbeddingChunker
from pipeline.components.chunking.recursive import RecursiveCharacterSplitter
# Embedding
from pipeline.components.bge_m3_embedders import BGEM3HybridDocumentEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# Document Store & Writer
from hypster import instantiate
from qdrant_client import QdrantClient
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from config.hypster_config import pipeline_config, EMBEDDER_OPTIONS, LLM_OPTIONS

load_dotenv()

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx"}
_BGE_M3 = "BAAI/bge-m3"


def _make_chunker(chunker_name: str, split_length: int | None = None, split_overlap: int | None = None, max_length: int | None = None, min_length: int | None = None):
  if chunker_name == "RecursiveCharacterSplitter":
    return RecursiveCharacterSplitter(split_length=split_length, split_overlap=split_overlap)
  if chunker_name == "FixedSizeTokenSplitter":
    return FixedSizeTokenSplitter(split_length=split_length, split_overlap=split_overlap)
  if chunker_name == "SemanticEmbeddingChunker":
    kwargs = {"max_length": max_length}
    if min_length is not None:
      kwargs["min_length"] = min_length
    return SemanticEmbeddingChunker(**kwargs)
  raise ValueError(f"Unknown chunker: {chunker_name}")


def _collection_name(chunker_name: str, embedder_model: str, llm_name: str | None = None) -> str:
  suffix = f"_{llm_name}" if llm_name else ""
  return f"{chunker_name}_{embedder_model}{suffix}".replace("/", "-").lower()


def _make_doc_embedder(emb_cfg: dict):
  if emb_cfg["api_model"] == _BGE_M3:
    return BGEM3HybridDocumentEmbedder()
  truncate_dim = emb_cfg.get("truncate_dim")
  return SentenceTransformersDocumentEmbedder(
    model=emb_cfg["api_model"],
    prefix=emb_cfg.get("doc_prefix", ""),
    batch_size=4,
    truncate_dim=truncate_dim,
  )


def _make_converter(unstructured_url: str) -> UnstructuredFileConverter:
  return UnstructuredFileConverter(
    api_url=unstructured_url,
    document_creation_mode="one-doc-per-page",
    unstructured_kwargs={
      "strategy": "hi_res",
      "languages": ["ita", "eng"],
      "skip_infer_table_types": [],
      "pdf_infer_table_structure": True,
    },
  )


def _free_semantic_chunker_gpu(chunker: SemanticEmbeddingChunker) -> None:
  """Move the semantic chunker's internal embedding model to CPU and clear CUDA cache."""
  try:
    import torch
    if not torch.cuda.is_available():
      return
    backend = getattr(getattr(chunker, 'embedder', None), 'embedding_backend', None)
    if backend is not None:
      model = getattr(backend, 'model', None)
      if model is not None:
        model.cpu()
    torch.cuda.empty_cache()
  except Exception:
    pass


_NON_SEMANTIC_CHUNKERS = ["RecursiveCharacterSplitter", "FixedSizeTokenSplitter"]
_SEMANTIC_CHUNKER = "SemanticEmbeddingChunker"


def run_indexing(resume_from: int = 0) -> None:
  non_semantic_combos = [(c, e, None) for c, e in product(_NON_SEMANTIC_CHUNKERS, EMBEDDER_OPTIONS)]
  semantic_combos = [(_SEMANTIC_CHUNKER, e, l) for e, l in product(EMBEDDER_OPTIONS, LLM_OPTIONS)]
  all_combos = non_semantic_combos + semantic_combos  # 6 + 9 = 15

  client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
  client.get_collections()

  if resume_from == 0:
    print("Clearing entire Qdrant database...")
    for c in client.get_collections().collections:
      print(f"  Deleting collection: {c.name}")
      client.delete_collection(c.name)
    print("Database cleared.\n")
  else:
    chunker_r, embedder_r, llm_r = all_combos[resume_from]
    llm_str = f" | {llm_r}" if llm_r else ""
    print(f"Resuming from combination {resume_from + 1}/{len(all_combos)}: {chunker_r} | {embedder_r}{llm_str}\n")

  raw_data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
  all_file_paths = [
    str(p) for p in raw_data_dir.iterdir()
    if p.suffix.lower() in _SUPPORTED_EXTENSIONS
  ]

  if not all_file_paths:
    print("No supported files found in /data/raw!")
    return

  print(f"Found {len(all_file_paths)} files to index: {[Path(p).name for p in all_file_paths]}\n")

  unstructured_url = os.getenv("UNSTRUCTURED_API_URL", "http://localhost:8000/general/v0/general")

  for idx, (chunker_name, embedder_model, llm_name) in enumerate(all_combos):
    if idx < resume_from:
      continue

    overrides = {
      "chunking.chunker_name": chunker_name,
      "embedding.model": embedder_model,
      "llm.name": llm_name or LLM_OPTIONS[0],
    }
    config = instantiate(pipeline_config, values=overrides, on_unknown="raise")
    emb_cfg = config["embedding"]
    llm_cfg = config["llm"]
    chunking_cfg = config["chunking"]

    llm_str = f" | {llm_name}" if llm_name else ""
    config_name = f"{chunker_name} | {embedder_model}{llm_str}"
    print(f"Indexing config ({idx + 1}/{len(all_combos)}): {config_name}")

    collection_name = _collection_name(chunker_name, embedder_model, llm_name)
    use_sparse = embedder_model == _BGE_M3

    document_store = QdrantDocumentStore(
      url=os.getenv("QDRANT_URL"),
      api_key=Secret.from_env_var("QDRANT_API_KEY"),
      index=collection_name,
      embedding_dim=emb_cfg["dims"],
      use_sparse_embeddings=use_sparse,
      recreate_index=True,
      timeout=300,
      write_batch_size=50,
      wait_result_from_api=False,
    )

    split_length = chunking_cfg.get("split_length")
    split_overlap = chunking_cfg.get("split_overlap")
    max_char_length = llm_cfg.get("max_char_length") if llm_name else None
    min_char_length = llm_cfg.get("min_char_length") if llm_name else None
    chunker = _make_chunker(chunker_name, split_length=split_length, split_overlap=split_overlap, max_length=max_char_length, min_length=min_char_length)
    embedder = _make_doc_embedder(emb_cfg)

    if chunker_name == "SemanticEmbeddingChunker":
      # Phase 1: chunk documents — the semantic chunker loads its embedding model onto GPU
      chunk_pipe = Pipeline()
      chunk_pipe.add_component("converter", _make_converter(unstructured_url))
      chunk_pipe.add_component("cleaner", DocumentCleaner())
      chunk_pipe.add_component("chunker", chunker)
      chunk_pipe.add_component("meta_cleaner", ChunkMetaCleaner())
      chunk_pipe.connect("converter.documents", "cleaner.documents")
      chunk_pipe.connect("cleaner.documents", "chunker.documents")
      chunk_pipe.connect("chunker.documents", "meta_cleaner.documents")
      chunk_result = chunk_pipe.run({"converter": {"paths": all_file_paths}})
      chunks = chunk_result["meta_cleaner"]["documents"]

      # Free the chunker's GPU memory before loading the indexing embedding model,
      # so both models are never in VRAM simultaneously (prevents OOM on 8 GiB GPUs).
      _free_semantic_chunker_gpu(chunker)
      del chunk_pipe
      gc.collect()

      # Phase 2: embed and write with GPU now free of the chunker model
      embed_pipe = Pipeline()
      embed_pipe.add_component("embedder", embedder)
      embed_pipe.add_component("writer", DocumentWriter(document_store=document_store))
      embed_pipe.connect("embedder.documents", "writer.documents")
      embed_pipe.run({"embedder": {"documents": chunks}})
    else:
      indexing_pipe = Pipeline()
      indexing_pipe.add_component("converter", _make_converter(unstructured_url))
      indexing_pipe.add_component("cleaner", DocumentCleaner())
      indexing_pipe.add_component("chunker", chunker)
      indexing_pipe.add_component("meta_cleaner", ChunkMetaCleaner())
      indexing_pipe.add_component("embedder", embedder)
      indexing_pipe.add_component("writer", DocumentWriter(document_store=document_store))
      indexing_pipe.connect("converter.documents", "cleaner.documents")
      indexing_pipe.connect("cleaner.documents", "chunker.documents")
      indexing_pipe.connect("chunker.documents", "meta_cleaner.documents")
      indexing_pipe.connect("meta_cleaner.documents", "embedder.documents")
      indexing_pipe.connect("embedder.documents", "writer.documents")
      indexing_pipe.run({"converter": {"paths": all_file_paths}})

    print(f"Finished indexing for {config_name}")

  print("Finished indexing pipeline!")


if __name__ == "__main__":
  setup_logging("indexing_pipeline")
  run_indexing()
