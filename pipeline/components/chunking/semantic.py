import dataclasses
from typing import List
from haystack import component, Document
from transformers import logging as hf_logging
from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter, RecursiveDocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder


@component
class SemanticEmbeddingChunker:
  """
  Semantic chunker that uses sentence-embedding similarity to find natural topic boundaries.
  Wraps Haystack's EmbeddingBasedDocumentSplitter and adds a character-level fallback splitter
  for chunks that still exceed max_length — mainly Excel-derived text where sentence
  tokenisation is unreliable. The outer component is required because EmbeddingBasedDocumentSplitter
  does not expose a fallback natively.

  GPU note: warm_up() loads the sentence-transformer model onto GPU. Call _free_semantic_chunker_gpu()
  in indexing_pipeline before loading the indexing embedder to avoid VRAM OOM.
  """
  def __init__(
    self,
    max_length: int,
    min_length: int,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    sentences_per_group: int = 2,
    percentile: float = 0.95,
    language: str = "it",
  ):
    hf_logging.set_verbosity_error()
    self.max_length = max_length

    self.embedder = SentenceTransformersDocumentEmbedder(
      model=model_name,
      progress_bar=False,
    )
    self.splitter = EmbeddingBasedDocumentSplitter(
      document_embedder=self.embedder,
      sentences_per_group=sentences_per_group,
      percentile=percentile,
      min_length=min_length,
      max_length=max_length,
      language=language,
    )

    self._fallback = RecursiveDocumentSplitter(
      separators=["\n\n", "\n", ". ", " "],
      split_length=self.max_length,
      split_overlap=0,
      split_unit="char"
    )

  def warm_up(self) -> None:
    self.splitter.warm_up()

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]) -> dict:
    chunks = self.splitter.run(documents=documents)["documents"]
    final: List[Document] = []

    for doc in chunks:
      if len(doc.content or "") > self.max_length:
        sub_chunks = self._fallback.run(documents=[doc])["documents"]
        for sub in sub_chunks:
          final.append(dataclasses.replace(sub, meta={**doc.meta, **sub.meta}))
      else:
        final.append(doc)

    return {"documents": final}
