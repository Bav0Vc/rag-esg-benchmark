"""
Haystack components for pre- and post-chunking document cleanup.

DocumentCleaner:   merges per-page documents from UnstructuredFileConverter into one document
                   per source file and injects <<<PAGE X>>> markers before each paragraph.
ChunkMetaCleaner:  runs after the chunker — extracts page metadata from the first marker in
                   each chunk, strips all markers from chunk text, and deduplicates.
"""
import re
import dataclasses
from pathlib import Path
from collections import defaultdict
from haystack import component, Document

_PAGE_MARKER = "<<<PAGE {}>>>"
_PAGE_MARKER_RE = re.compile(r"<<<PAGE ([^>]+)>>>")
# Chunks split mid-marker produce two fragments: "<<<PAGE" (open half) and "14>>>" (close half).
_STRAY_OPEN_RE = re.compile(r"<<<PAGE[^>]*>*")  # "<<<PAGE", "<<<PAGE 14", "<<<PAGE 14>>" etc.
_STRAY_CLOSE_RE = re.compile(r"^\d+>>>\s*")      # "14>>>" at the very start of a chunk


@component
class ChunkMetaCleaner:
  """
  Runs after chunking.
  - Extracts page from the first <<<PAGE X>>> marker found in the chunk text.
    If no marker is present the chunk is mid-page and inherits the parent doc's page.
  - Strips all markers from the chunk text before embedding.
  - Deduplicates chunks by content (handles duplicate overlap chunks from short pages).
  """
  @component.output_types(documents=list[Document])
  def run(self, documents: list[Document]) -> dict:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
      content = doc.content or ""

      markers = _PAGE_MARKER_RE.findall(content)
      clean = _PAGE_MARKER_RE.sub("", content)
      clean = _STRAY_OPEN_RE.sub("", clean)
      clean = _STRAY_CLOSE_RE.sub("", clean)
      clean = clean.strip()

      if not clean or clean in seen:
        continue
      seen.add(clean)

      meta = {k: v for k, v in doc.meta.items() if k != "page_number"}
      if markers:
        meta["page"] = markers[0]
      unique.append(dataclasses.replace(doc, content=clean, meta=meta))

    return {"documents": unique}


@component
class DocumentCleaner:
  """
  Runs after UnstructuredFileConverter.
  - Drops empty documents.
  - Normalises whitespace while preserving Markdown table formatting for Excel.
  - Merges all pages of the same file into a single Document, inserting
    <<<PAGE X>>> markers between pages so chunkers can produce cross-page
    chunks while ChunkMetaCleaner can still recover which page each chunk is on.
  """

  @component.output_types(documents=list[Document])
  def run(self, documents: list[Document]) -> dict:
    pages_by_source: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
      source = (doc.meta or {}).get("filename") or Path((doc.meta or {}).get("file_path", "")).name
      text = self._clean_text(doc.content or "", source)
      if not text:
        continue
      meta = self._build_meta(doc.meta)
      pages_by_source[meta["source"]].append({
        "text": text,
        "page": meta["page"],
        "source": meta["source"],
      })

    merged: list[Document] = []
    for source, pages in pages_by_source.items():
      pages.sort(key=lambda p: (0, int(str(p["page"]))) if str(p["page"]).isdigit() else (1, str(p["page"])))
      parts = [self._mark_paragraphs(p["text"], p["page"]) for p in pages]
      merged_text = "\n\n".join(parts)
      merged.append(Document(
        content=merged_text,
        meta={"source": source, "page": pages[0]["page"]},
      ))

    return {"documents": merged}

  def _mark_paragraphs(self, text: str, page: str) -> str:
    """Prefix every paragraph with <<<PAGE X>>> so every chunk contains a page marker."""
    marker = _PAGE_MARKER.format(page)
    paragraphs = re.split(r"\n\n+", text)
    marked = [f"{marker}\n{p.strip()}" for p in paragraphs if p.strip()]
    return "\n\n".join(marked)

  def _clean_text(self, text: str, source: str = "") -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    if source.lower().endswith((".xlsx", ".xls")):
      text = re.sub(r" {2,}", " ", text)
    else:
      text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

  def _build_meta(self, original_meta: dict) -> dict:
    meta = dict(original_meta)
    if "filename" in meta:
      meta["source"] = meta.pop("filename")
    elif "file_path" in meta:
      meta["source"] = Path(meta["file_path"]).name
    meta.pop("file_path", None)

    source = meta.get("source", "")
    if source.lower().endswith((".xlsx", ".xls")) and "page_name" in meta:
      meta["page"] = meta.pop("page_name")
    elif "page_number" in meta:
      meta["page"] = meta.pop("page_number")
    else:
      meta.setdefault("page", "?")

    return meta
