"""
Query pipeline: for each pipeline configuration, retrieves relevant chunks from Qdrant and generates
an Italian JSON answer via the configured LLM.

BGE-M3 uses QdrantHybridRetriever (dense + sparse via RRF); all other embedders use
QdrantEmbeddingRetriever (dense only). Transient API failures are retried with exponential backoff
(up to 6 attempts, base delay 15 s).
"""
import os
import time
from haystack import Pipeline
from dotenv import load_dotenv
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from pipeline.components.bge_m3_embedders import BGEM3HybridTextEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever, QdrantHybridRetriever

load_dotenv()


_PROMPT_TEMPLATE = """Answer the question based on the context. \
Also include the filename and page number or page name of the document containing the retrieved chunk, on a new line after the answer to the question. Output only in valid JSON.
Context:
{% for doc in documents %}
  File: {{ doc.meta['source'] }}, Page: {{ doc.meta.get('page', '?') }}
  Contents: {{ doc.content }}
{% endfor %}
Question: {{question}}"""


def _extract_reply_text(reply) -> str:
  if hasattr(reply, "content"):
    if isinstance(reply.content, list):
      return "\n".join(
        c.text if getattr(c, "text", None) is not None else str(c)
        for c in reply.content
      )
    return reply.content if isinstance(reply.content, str) else str(reply.content)
  if hasattr(reply, "text"):
    return reply.text
  return str(reply)

def _build_llm(llm_cfg: dict):
  if llm_cfg["backend"] == "mistral":
    return MistralChatGenerator(model=llm_cfg["api_model"])
  if llm_cfg["backend"] == "hf":
    return OpenAIGenerator(model=llm_cfg["api_model"], api_key=Secret.from_env_var("HF_TOKEN"), api_base_url=llm_cfg["api_base_url"])


def run_query_pipeline(config: dict, golden_set: list) -> list:
  chunker_name: str = config["chunking"]["chunker_name"]
  emb_cfg: dict = config["embedding"]
  llm_cfg: dict = config["llm"]

  embedder_model = emb_cfg["model"]
  llm_name = llm_cfg["name"]
  config_label = f"{chunker_name} | {embedder_model} | {llm_name}"
  print(f"Running config: {config_label}")

  llm_suffix = f"_{llm_name}" if chunker_name == "SemanticEmbeddingChunker" else ""
  collection_name = f"{chunker_name}_{embedder_model}{llm_suffix}".replace("/", "-").lower()
  use_hybrid = embedder_model == "BAAI/bge-m3"

  document_store = QdrantDocumentStore(
    url=os.getenv("QDRANT_URL"),
    api_key=Secret.from_env_var("QDRANT_API_KEY"),
    index=collection_name,
    embedding_dim=emb_cfg["dims"],
    use_sparse_embeddings=use_hybrid,
    recreate_index=False,
  )

  try:
    count = document_store.count_documents()
    if count == 0:
      print(f"  -> Skipping. Collection '{collection_name}' is empty.")
      return []
  except Exception as exc:
    print(f"  -> Could not connect to collection '{collection_name}'. Skipping. {exc}")
    return []

  llm_instance = _build_llm(llm_cfg)
  query_pipe = Pipeline()

  if use_hybrid:
    # Single component produces both dense and sparse — 1 encode() call for BGE-M3.
    query_pipe.add_component("text_embedder", BGEM3HybridTextEmbedder(query_instruction=emb_cfg.get("query_prefix")))
    query_pipe.add_component("retriever", QdrantHybridRetriever(document_store=document_store))
    query_pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipe.connect("text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
  else:
    truncate_dim = emb_cfg.get("truncate_dim")
    query_pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model=emb_cfg["api_model"], prefix=emb_cfg.get("query_prefix", ""), truncate_dim=truncate_dim))
    query_pipe.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
    query_pipe.connect("text_embedder.embedding", "retriever.query_embedding")

  query_pipe.add_component("prompt_builder", PromptBuilder(template=_PROMPT_TEMPLATE, required_variables=["documents", "question"]))
  query_pipe.add_component("llm", llm_instance)
  query_pipe.connect("retriever.documents", "prompt_builder.documents")
  query_pipe.connect("prompt_builder", "llm")

  _MAX_RETRIES = 6
  _RETRY_BASE_DELAY = 15

  results = []
  n_questions = len(golden_set)
  for q_idx, item in enumerate(golden_set, start=1):
    q = item["question"]
    for attempt in range(_MAX_RETRIES):
      try:
        start_time = time.time()
        response = query_pipe.run(
          {"text_embedder": {"text": q}, "prompt_builder": {"question": q}},
          include_outputs_from={"retriever"},
        )
        latency = time.time() - start_time

        reply = response["llm"]["replies"][0]
        final_answer = _extract_reply_text(reply)

        contexts = [doc.content for doc in response["retriever"]["documents"]]
        if hasattr(reply, "meta"):
          usage = reply.meta.get("usage", {})
        else:
          meta_list = response["llm"].get("meta", [{}])
          usage = (meta_list[0].get("usage", {}) if meta_list else {})

        results.append({
          "question_id": item["question_id"],
          "question": q,
          "ground_truth": item["ground_truth"],
          "expected_source": item["expected_source"],
          "reference_contexts": list(item.get("reference_contexts", {}).values()),
          "source_page": item.get("source_page"),
          "acceptable_sources": item.get("acceptable_sources", []),
          "Configuration": config_label,
          "Chunker": chunker_name,
          "Embedder": embedder_model,
          "LLM": llm_name,
          "contexts": contexts,
          "answer": final_answer,
          "latency": latency,
          "prompt_tokens": usage.get("prompt_tokens", 0),
          "completion_tokens": usage.get("completion_tokens", 0),
        })
        print(f"  [{q_idx}/{n_questions}] Q{item['question_id']} done ({latency:.1f}s)")
        break
      except Exception as exc:
        if attempt < _MAX_RETRIES - 1:
          wait = _RETRY_BASE_DELAY * (2 ** attempt)
          print(f"[attempt {attempt + 1}/{_MAX_RETRIES}] '{q}' failed: {exc}.\nRetrying in {wait}s...")
          remaining = float(wait)
          while remaining > 0:
            sleep_for = min(remaining, 5.0)
            time.sleep(sleep_for)
            remaining -= sleep_for
            print(f"Retrying in {remaining:.0f}s...")
        else:
          raise RuntimeError(
            f"Question '{q}' (config: {config_label}) failed after {_MAX_RETRIES} attempts."
            f"Last error: {exc}"
          ) from exc

  print(f"  -> Finished {len(results)}/{n_questions} questions for {config_label}")
  return results
