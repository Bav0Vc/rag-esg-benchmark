# rag-esg-benchmark

A research benchmark evaluating **27 RAG pipeline configurations** (3 chunkers × 3 embedders × 3 LLMs) for ESG document compliance analysis on Italian-language corporate documents.

The pipeline indexes documents into [Qdrant](https://qdrant.tech/), answers a 20-question golden-set questionnaire, and scores answers with [RAGAS](https://docs.ragas.io/) metrics (Faithfulness, Context Recall, Context Precision, Answer Relevancy) plus a custom Source Attribution metric.

---

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with CUDA 12.4+ (required for local embedding models — RTX 30/40 series recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) configured for Docker
- External API accounts: [Qdrant Cloud](https://cloud.qdrant.io/), [HuggingFace](https://huggingface.co/), [Mistral AI](https://mistral.ai/)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Bav0Vc/rag_esg_benchmark.git
cd rag_esg_benchmark
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

See [Environment Variables](#environment-variables) for the full list.

### 3. Add source documents

Place raw ESG documents (PDF, DOCX, XLSX) in `data/raw/` and a golden-set questionnaire in `data/golden_set/questionnaire.json`.  
See [`data/golden_set/questionnaire.json`](data/golden_set/questionnaire.json) for the expected schema.

### 4. Start the containers

```bash
docker-compose up -d
```

Two containers start:

| Container          | Role                                                           |
| ------------------ | -------------------------------------------------------------- |
| `unstructured-api` | Local Unstructured API — converts PDFs/DOCX to structured text |
| `app`              | RAG pipeline — indexing, querying, evaluation                  |

### 5. Pre-download embedding models (first run only)

```bash
docker exec -it rag_esg_benchmark-app-1 python scripts/download_models.py
```

This downloads all three embedding models into a persistent HuggingFace cache volume.

---

## Environment Variables

| Variable         | Required | Description                                                               |
| ---------------- | -------- | ------------------------------------------------------------------------- |
| `QDRANT_URL`     | Yes      | Qdrant cluster URL (e.g. `https://xyz.qdrant.io`)                         |
| `QDRANT_API_KEY` | Yes      | Qdrant API key                                                            |
| `HF_TOKEN`       | Yes      | HuggingFace access token (embedding models + LLM inference via HF router) |
| `RAGAS_BASE_URL` | Yes      | OpenAI-compatible endpoint for the RAGAS evaluator LLM                    |
| `RAGAS_MODEL`    | Yes      | Model name for the RAGAS evaluator (e.g. `gpt-4o-mini`)                   |
| `SCHOOL_API_KEY` | No       | Alternative key for the RAGAS evaluator — overrides `HF_TOKEN` if set     |

---

## Running the Pipeline

### Full end-to-end run

```bash
docker exec -it rag_esg_benchmark-app-1 python -m orchestration.run_pipeline
```

### Individual stages

```bash
# Stage 1 — Index all chunker × embedder combinations into Qdrant
docker exec -it rag_esg_benchmark-app-1 python -m pipeline.indexing_pipeline

# Stage 2 — Benchmark query loop across all 27 configurations
docker exec -it rag_esg_benchmark-app-1 python -m orchestration.benchmark_loop

# Stage 3 — Score answers with RAGAS metrics
docker exec -it rag_esg_benchmark-app-1 python -m evaluation.ragas_eval
```

### Resume indexing after interruption

```bash
# Resume from combination index 6 (0-based)
docker exec -it rag_esg_benchmark-app-1 python -c \
  "from pipeline.indexing_pipeline import run_indexing; run_indexing(resume_from=6)"
```

### Rebuild after dependency changes

```bash
docker compose build app && docker compose up -d app
```

---

## Directory Structure

```
rag_esg_benchmark/
├── config/
│   └── hypster_config.py         # All 27 pipeline variants (hypster-based config)
├── pipeline/
│   ├── indexing_pipeline.py      # Document conversion → chunking → embedding → Qdrant
│   ├── query_pipeline.py         # Per-configuration question answering
│   └── components/
│       ├── document_cleaner.py           # PAGE marker injection & text normalisation
│       ├── bge_m3_embedders.py           # Custom BGE-M3 dense+sparse Haystack components
│       └── chunking/
│           ├── fixed.py                  # FixedSizeTokenSplitter
│           ├── recursive.py              # RecursiveCharacterSplitter
│           └── semantic.py               # SemanticEmbeddingChunker
├── orchestration/
│   ├── run_pipeline.py           # Full pipeline orchestration entry point
│   └── benchmark_loop.py         # Iterates all 27 configs, calls query_pipeline
├── evaluation/
│   ├── ragas_eval.py             # RAGAS scoring + source attribution
│   └── results/                  # Output files (gitignored)
│       ├── evaluation_dataset.json
│       ├── per_question_scores.json
│       └── metrics_leaderboard.csv
├── scripts/
│   ├── download_models.py        # Pre-download HuggingFace embedding models
│   └── clear_qdrant.py           # Delete Qdrant collections
├── data/                         # Gitignored — provide your own documents
│   ├── raw/                      # Source ESG documents (PDF, DOCX, XLSX)
│   └── golden_set/
│       └── questionnaire.json    # 20-question evaluation set with ground truth
├── .env.example                  # Environment variable template
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Configuration

All pipeline variants are defined in [`config/hypster_config.py`](config/hypster_config.py) using [hypster](https://github.com/gilad-rubin/hypster).

**Benchmarked components:**

| Axis     | Options                                                                                             |
| -------- | --------------------------------------------------------------------------------------------------- |
| Chunker  | `RecursiveCharacterSplitter`, `FixedSizeTokenSplitter`, `SemanticEmbeddingChunker`                  |
| Embedder | `BAAI/bge-m3`, `Snowflake/snowflake-arctic-embed-l-v2.0`, `intfloat/multilingual-e5-large-instruct` |
| LLM      | `Gemma-3-27b-it`, `Llama-3.3-70B-Instruct`, `Mistral-Small-2603`                                    |

Fixed hyperparameters: chunk size = 1024 tokens, overlap = 154 tokens (~15%), top-k = 3 retrieved chunks.

Qdrant collection names follow the pattern `{chunker}_{embedder}[_{llm}]` (slashes → dashes, lowercased).

---

## Evaluation Metrics

| Metric             | Source                               | Weight in S_overall |
| ------------------ | ------------------------------------ | ------------------- |
| Faithfulness       | RAGAS 0.4.3                          | 0.35                |
| Source Attribution | Custom (exact filename + page match) | 0.25                |
| Context Recall     | RAGAS 0.4.3                          | 0.15                |
| Answer Relevancy   | RAGAS 0.4.3                          | 0.15                |
| Context Precision  | RAGAS 0.4.3                          | 0.10                |

S_overall = 0.35·F + 0.15·AR + 0.25·SA + 0.15·CR + 0.10·CP

---

## Data & Privacy

Source documents and the golden-set questionnaire are **not included**. The benchmark was developed using Italian ESG compliance documents provided under a research agreement; these documents are not redistributed.
