"""
Benchmark loop: iterates all 27 pipeline configurations, runs each against the golden-set
questionnaire via run_query_pipeline(), and writes results incrementally to
evaluation/results/evaluation_dataset.json. Already-completed configurations are skipped,
so the loop is safe to restart after interruption.
"""
import os
import json
import pandas as pd
from itertools import product
from hypster import instantiate
from scripts.logger import setup_logging
from pipeline.query_pipeline import run_query_pipeline
from config.hypster_config import pipeline_config, CHUNKER_OPTIONS, EMBEDDER_OPTIONS, LLM_OPTIONS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output paths
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "results")
EVAL_DATASET_PATH = os.path.join(_RESULTS_DIR, "evaluation_dataset.json")
BENCHMARK_CSV_PATH = os.path.join(_RESULTS_DIR, "rag_benchmark_results.csv")


def load_golden_dataset() -> list:
  path = os.path.join(_PROJECT_ROOT, "data", "golden_set", "questionnaire.json")
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

def _load_existing_results() -> list[dict]:
  if not os.path.exists(EVAL_DATASET_PATH):
    return []
  with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
    return json.load(f)


def run_benchmark() -> None:
  os.makedirs(_RESULTS_DIR, exist_ok=True)

  golden_dataset = load_golden_dataset()
  combinations = list(product(CHUNKER_OPTIONS, EMBEDDER_OPTIONS, LLM_OPTIONS))

  all_results = _load_existing_results()
  done_configs = {row["Configuration"] for row in all_results}
  if done_configs:
    print(f"Resuming: {len(done_configs)}/{len(combinations)} configurations already completed.")
  print(f"Starting benchmark: {len(combinations)} configurations × {len(golden_dataset)} questions.\n")

  for idx, (chunker_name, embedder_model, llm_name) in enumerate(combinations, start=1):
    config_label = f"{chunker_name} | {embedder_model} | {llm_name}"
    if config_label in done_configs:
      print(f"[{idx}/{len(combinations)}] Skipping (done): {config_label}")
      continue

    overrides = {
      "chunking.chunker_name": chunker_name,
      "embedding.model": embedder_model,
      "llm.name": llm_name,
    }
    config = instantiate(pipeline_config, values=overrides, on_unknown="raise")

    print(f"[{idx}/{len(combinations)}] {chunker_name} | {embedder_model} | {llm_name}")
    results = run_query_pipeline(config, golden_dataset)
    all_results.extend(results)

  with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
  print(f"\nSaved {len(all_results)} rows to {EVAL_DATASET_PATH}")

  df = pd.DataFrame(all_results)
  df.to_csv(BENCHMARK_CSV_PATH, index=False, sep=";")
  print(f"Saved benchmark CSV to {BENCHMARK_CSV_PATH}")


if __name__ == "__main__":
  setup_logging("benchmark_loop")
  run_benchmark()
