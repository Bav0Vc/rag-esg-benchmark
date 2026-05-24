from typing import List
from haystack import component, Document
from haystack.components.preprocessors import RecursiveDocumentSplitter


@component
class FixedSizeTokenSplitter:
  """
  Fixed-width token-window splitter. Separators [" ", ""] split only on word boundaries,
  never on newlines — PAGE markers that fall mid-chunk are left intact for ChunkMetaCleaner
  to strip. Use RecursiveCharacterSplitter instead when paragraph-boundary splits matter.
  """
  def __init__(self, split_length: int, split_overlap: int):
    self.internal_splitter = RecursiveDocumentSplitter(
      separators=[" ", ""],
      split_length=split_length,
      split_overlap=split_overlap,
      split_unit="token",
    )

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.internal_splitter.run(documents=documents)
