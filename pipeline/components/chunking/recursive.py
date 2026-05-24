from typing import List
from haystack import component, Document
from haystack.components.preprocessors import RecursiveDocumentSplitter


@component
class RecursiveCharacterSplitter:
  """
  Paragraph-aware token splitter. The default separator hierarchy ["\n\n", "\n", ". ", " ", ""]
  splits preferentially at paragraph breaks, which is where PAGE markers are injected — so chunks
  are more likely to start with a clean <<<PAGE X>>> marker and carry accurate page metadata.
  """
  def __init__(self, split_length: int, split_overlap: int, separators: List[str] = ["\n\n", "\n", ". ", " ", ""]):
    self.internal_splitter = RecursiveDocumentSplitter(
      separators=separators,
      split_length=split_length,
      split_overlap=split_overlap,
      split_unit="token",
    )

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.internal_splitter.run(documents=documents)
