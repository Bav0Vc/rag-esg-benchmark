"""
RAGAS evaluation: scores each answer in evaluation_dataset.json with four RAGAS 0.4.3 metrics
(Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy) using an async OpenAI-compatible
LLM evaluator. Source Attribution is computed separately — not via RAGAS — by checking whether
the expected filename and page number appear in the answer string.

Writes results incrementally after each row to avoid losing progress on API interruptions.
Produces evaluation/results/per_question_scores.json and evaluation/results/metrics_leaderboard.csv.
"""
import os
import json
import asyncio
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from scripts.logger import setup_logging
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy)


load_dotenv()

# region Constants
_RAGAS_METRICS = ["faithfulness", "context_recall", "context_precision", "answer_relevancy"]
_LEADERBOARD_METRIC_COLS = {
  "faithfulness": "Faithfulness",
  "context_recall": "Context Recall",
  "context_precision": "Context Precision",
  "answer_relevancy": "Answer Relevancy",
}
_META_COLS = ["question_id", "Configuration", "Chunker", "Embedder", "LLM", "latency", "source_attribution", "prompt_tokens", "completion_tokens"]

_EVAL_BASE_URL = os.environ["RAGAS_BASE_URL"]
_EVAL_API_KEY_ENV = "SCHOOL_API_KEY" if os.environ.get("SCHOOL_API_KEY") else "HF_TOKEN"
_EVAL_MODEL = os.environ["RAGAS_MODEL"]
# endregion


def _extract_answer_text(raw: str) -> str:
  """Extract plain answer text from the two answer formats used in evaluation_dataset.json."""
  raw = raw.strip()
  if raw.startswith("```"):
    inner = raw.strip("`").strip()
    if inner.startswith("json"):
      inner = inner[4:].strip()
    try:
      return json.loads(inner).get("answer", inner)
    except Exception:
      return inner
  try:
    parsed = json.loads(raw.split("\n")[0])
    return parsed.get("risposta") or parsed.get("answer") or raw
  except Exception:
    return raw


# region Per-sample RAGAS-metrics
async def score_sample(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, row):
  user_input = str(row.get("question", ""))
  response = _extract_answer_text(str(row.get("answer", ""))) if row.get("answer") else ""
  retrieved_contexts = [str(c) for c in (row.get("contexts") or [])][:3]
  reference = str(row.get("ground_truth", "")) if row.get("ground_truth") else ""

  async def _faithfulness():
    try:
      result = await faithfulness_m.ascore(user_input=user_input, response=response, retrieved_contexts=retrieved_contexts)
      return result.value
    except Exception as e:
      print(f"Faithfulness failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  async def _context_recall():
    try:
      result = await context_recall_m.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference)
      return result.value
    except Exception as e:
      print(f"ContextRecall failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  async def _context_precision():
    try:
      result = await context_precision_m.ascore(user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts)
      return result.value
    except Exception as e:
      print(f"ContextPrecision failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  async def _answer_relevancy():
    try:
      result = await answer_relevancy_m.ascore(user_input=user_input, response=response)
      return result.value
    except Exception as e:
      print(f"AnswerRelevancy failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  faith, recall, precision, relevancy = await asyncio.gather(_faithfulness(), _context_recall(), _context_precision(), _answer_relevancy())

  return {
    "faithfulness": faith,
    "context_recall": recall,
    "context_precision": precision,
    "answer_relevancy": relevancy,
  }
# endregion


# region NaN RAGAS-metric retry
async def _retry_failed_metrics(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, per_question_file, leaderboard_file, eval_df):
  with open(per_question_file, encoding="utf-8") as f:
    pq_data = json.load(f)

  to_retry = [
    (i, metric)
    for i, r in enumerate(pq_data)
    for metric in _RAGAS_METRICS
    if r.get(metric) != r.get(metric)
  ]

  if not to_retry:
    print("\nNo NaN metric scores to retry.")
    return

  print(f"\nRetrying {len(to_retry)} NaN metric score(s)...")
  updated_configs: set[str] = set()

  for idx, metric in to_retry:
    row = pq_data[idx]
    config = row["Configuration"]
    question_id = row["question_id"]

    match = eval_df[(eval_df["Configuration"] == config) & (eval_df["question_id"] == question_id)]
    if match.empty:
      print(f"  ! Could not find original data for {config} | {question_id}, skipping.")
      continue

    orig = match.iloc[0]
    user_input = str(orig.get("question", ""))
    response = str(orig.get("answer", "")) if orig.get("answer") else ""
    retrieved_contexts = [str(c) for c in (orig.get("contexts") or [])]
    reference = str(orig.get("ground_truth", "")) if orig.get("ground_truth") else ""

    scorers = {
      "faithfulness":      lambda: faithfulness_m.ascore(user_input=user_input, response=response, retrieved_contexts=retrieved_contexts),
      "context_recall":    lambda: context_recall_m.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference),
      "context_precision": lambda: context_precision_m.ascore(user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts),
      "answer_relevancy":  lambda: answer_relevancy_m.ascore(user_input=user_input, response=response),
    }

    print(f"  Retrying {metric} for {config} | {question_id} ...")
    try:
      result = await scorers[metric]()
      pq_data[idx][metric] = result.value
      print(f"  -> {metric} = {result.value:.4f}")
      updated_configs.add(config)
    except Exception as e:
      print(f"  ! Still failed: {type(e).__name__}: {e}")

  with open(per_question_file, "w", encoding="utf-8") as f:
    json.dump(pq_data, f, ensure_ascii=False, indent=2)
  print(f"Saved updated {per_question_file}")

  if updated_configs:
    leaderboard_df = pd.read_csv(leaderboard_file, sep=";")
    pq_df = pd.DataFrame(pq_data)

    for config in updated_configs:
      subset = pq_df[pq_df["Configuration"] == config]
      mask = leaderboard_df["Configuration"] == config
      for metric, lb_col in _LEADERBOARD_METRIC_COLS.items():
        if lb_col in leaderboard_df.columns:
          new_mean = round(subset[metric].mean(), 4)
          leaderboard_df.loc[mask, lb_col] = new_mean
          print(f"  Updated leaderboard: {config} -> {lb_col} = {new_mean}")

    leaderboard_df.to_csv(leaderboard_file, index=False, sep=";")
    print(f"Saved updated {leaderboard_file}")
# endregion


# region Main evaluation loop
async def evaluate_results():
  input_file = "evaluation/results/evaluation_dataset.json"
  per_question_file = "evaluation/results/per_question_scores.json"
  leaderboard_file = "evaluation/results/metrics_leaderboard.csv"

  if not os.path.exists(input_file):
    print(f"Data not found at {input_file}. Run benchmark_loop.py first.")
    return

  with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

  df = pd.DataFrame(data)

  def _all_valid_sources(row) -> list[tuple[str, str]]:
    """Returns (file, page) pairs for the primary source and all acceptable alternatives."""
    primary = (str(row.get("expected_source", "")), str(row.get("source_page", "")))
    alternatives = [
      (str(s.get("file", "")), str(s.get("page", "")))
      for s in (row.get("acceptable_sources") or [])
    ]
    return [primary] + alternatives

  df["source_attribution"] = df.apply(
    lambda row: 1.0 if any(
      f and p and f in str(row.get("answer", "")) and p in str(row.get("answer", ""))
      for f, p in _all_valid_sources(row)
    ) else 0.0,
    axis=1,
  )

  # Load existing results so we can resume a previously interrupted run.
  os.makedirs("evaluation/results", exist_ok=True)

  existing_pq: list[dict] = []
  if os.path.exists(per_question_file):
    with open(per_question_file, "r", encoding="utf-8") as f:
      existing_pq = json.load(f)

  existing_leaderboard: list[dict] = []
  if os.path.exists(leaderboard_file):
    existing_leaderboard = pd.read_csv(leaderboard_file, sep=";").to_dict("records")

  already_done = {row["Configuration"] for row in existing_leaderboard}

  llm_client = AsyncOpenAI(base_url=_EVAL_BASE_URL, api_key=os.environ[_EVAL_API_KEY_ENV])
  
  evaluator_llm = llm_factory(
    _EVAL_MODEL,
    provider="openai",
    client=llm_client,
    max_tokens=16384,
    frequency_penalty=1.2,
    presence_penalty=0.5,
  )

  evaluator_llm.client = instructor.from_openai(llm_client, mode=instructor.Mode.MD_JSON)
  evaluator_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

  faithfulness_m = Faithfulness(llm=evaluator_llm)
  context_recall_m = ContextRecall(llm=evaluator_llm)
  context_precision_m = ContextPrecision(llm=evaluator_llm)
  answer_relevancy_m = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

  for config in df["Configuration"].unique():
    if config in already_done:
      print(f"Skipping {config} (already evaluated)")
      continue

    print(f"\nEvaluating: {config}")
    subset = df[df["Configuration"] == config].reset_index(drop=True)

    rows_scores = []
    for _, row in subset.iterrows():
      t0 = asyncio.get_event_loop().time()
      scores = await score_sample(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, row)
      elapsed = asyncio.get_event_loop().time() - t0
      print(f"  [{datetime.now().strftime('%H:%M:%S')}] | [{row['question_id']}] evaluated ({elapsed:.1f}s)")
      rows_scores.append(scores)

    scores_df = pd.DataFrame(rows_scores)
    for col in _META_COLS:
      if col in subset.columns:
        scores_df[col] = subset[col].values

    meta_present = [c for c in _META_COLS if c in scores_df.columns]
    ragas_present = [c for c in _RAGAS_METRICS if c in scores_df.columns]
    other = [c for c in scores_df.columns if c not in meta_present + ragas_present]
    scores_df = scores_df[meta_present + ragas_present + other]

    existing_pq.extend(scores_df.to_dict("records"))
    with open(per_question_file, "w", encoding="utf-8") as f:
      json.dump(existing_pq, f, ensure_ascii=False, indent=2)
    print(f"  Saved per-question scores ({len(existing_pq)} rows total)")

    def _mean(col):
      series = scores_df[col]
      return round(series.mean(), 4) if not series.isna().all() else np.nan

    existing_leaderboard.append({
      "Configuration": config,
      "Chunker": subset.iloc[0]["Chunker"],
      "Embedder": subset.iloc[0]["Embedder"],
      "LLM": subset.iloc[0]["LLM"],
      "Latency (s)": round(subset["latency"].mean(), 3),
      "Source Attribution": round(subset["source_attribution"].mean(), 2),
      "Faithfulness": _mean("faithfulness"),
      "Context Recall": _mean("context_recall"),
      "Context Precision": _mean("context_precision"),
      "Answer Relevancy": _mean("answer_relevancy"),
    })

    pd.DataFrame(existing_leaderboard).to_csv(leaderboard_file, index=False, sep=";")
    print(f"  Updated leaderboard ({len(existing_leaderboard)} configs so far)")

  await _retry_failed_metrics(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, per_question_file, leaderboard_file, df)

  print(f"\nSaved {len(existing_pq)} per-question score rows to {per_question_file}")
  print(f"Leaderboard saved to {leaderboard_file}")

  df_leaderboard = pd.DataFrame(existing_leaderboard)
  print("\n--- Leaderboard Summary ---")
  print(df_leaderboard.to_string())
# endregion

if __name__ == "__main__":
  setup_logging("ragas_eval")
  asyncio.run(evaluate_results())
