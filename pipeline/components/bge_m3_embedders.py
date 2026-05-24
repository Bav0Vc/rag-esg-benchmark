"""
Custom Haystack components for BAAI/bge-m3 that produce both dense and sparse vectors in one call.

The standard SentenceTransformersDocumentEmbedder only returns dense vectors, so these wrappers
use FlagEmbedding directly to support hybrid (dense + sparse) retrieval against Qdrant.
"""
import dataclasses
from typing import List, Optional
from haystack import component, Document
from haystack.dataclasses import SparseEmbedding


@component
class BGEM3HybridDocumentEmbedder:
  """
  Embeds documents with BAAI/bge-m3 using FlagEmbedding, producing both dense
  (doc.embedding) and sparse (doc.sparse_embedding) vectors in a single encode call.
  Compatible with QdrantDocumentStore(use_sparse_embeddings=True).
  """

  def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 4):
    self.model_name = model_name
    self.batch_size = batch_size
    self._model = None

  def warm_up(self):
    if self._model is not None:
      return
    try:
      from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
      raise ImportError(
        "FlagEmbedding is required for BGEM3HybridDocumentEmbedder. "
        "Add 'FlagEmbedding' to requirements.txt and rebuild the image."
      ) from exc
    try:
      import torch
      use_fp16 = torch.cuda.is_available()
    except ImportError:
      use_fp16 = False
    self._model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    if self._model is None:
      raise RuntimeError("warm_up() must be called before run().")

    texts = [doc.content or "" for doc in documents]

    result_docs = []
    for batch_start in range(0, len(texts), self.batch_size):
      batch_texts = texts[batch_start:batch_start + self.batch_size]
      batch_docs = documents[batch_start:batch_start + self.batch_size]

      output = self._model.encode_corpus(
        batch_texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
      )

      for doc, dense_vec, lex_weights in zip(batch_docs, output["dense_vecs"], output["lexical_weights"]):
        result_docs.append(dataclasses.replace(
          doc,
          embedding=dense_vec.tolist(),
          sparse_embedding=SparseEmbedding(
            indices=[int(k) for k in lex_weights.keys()],
            values=[float(v) for v in lex_weights.values()],
          ),
        ))

    return {"documents": result_docs}


@component
class BGEM3HybridTextEmbedder:
  """
  Embeds a query string with BAAI/bge-m3, returning both the dense embedding
  and the sparse embedding in a single encode call.
  Connects directly to QdrantHybridRetriever via query_embedding + query_sparse_embedding.
  """

  def __init__(self, model_name: str = "BAAI/bge-m3", query_instruction: Optional[str] = None):
    self.model_name = model_name
    self.query_instruction = query_instruction or None  # treat "" as None
    self._model = None

  def warm_up(self):
    if self._model is not None:
      return
    try:
      from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
      raise ImportError(
        "FlagEmbedding is required for BGEM3HybridTextEmbedder. "
        "Add 'FlagEmbedding' to requirements.txt and rebuild the image."
      ) from exc
    try:
      import torch
      use_fp16 = torch.cuda.is_available()
    except ImportError:
      use_fp16 = False
    self._model = BGEM3FlagModel(
      self.model_name,
      use_fp16=use_fp16,
      query_instruction_for_retrieval=self.query_instruction,
    )

  @component.output_types(embedding=List[float], sparse_embedding=SparseEmbedding)
  def run(self, text: str):
    if self._model is None:
      raise RuntimeError("warm_up() must be called before run().")

    output = self._model.encode_queries(
      [text],
      return_dense=True,
      return_sparse=True,
      return_colbert_vecs=False,
    )

    lex_weights = output["lexical_weights"][0]
    return {
      "embedding": output["dense_vecs"][0].tolist(),
      "sparse_embedding": SparseEmbedding(
        indices=[int(k) for k in lex_weights.keys()],
        values=[float(v) for v in lex_weights.values()],
      ),
    }
