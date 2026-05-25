# Keyword-Graph: Requirement Analysis Pipeline

A pipeline for analyzing software requirements through keyword extraction, semantic graph construction, clustering, and LLM-based inconsistency detection.

## Pipeline Overview

```
Phase 1: Keyword Extraction (KeyBERT or Gemini 2.5 Flash)
   ↓
Phase 2: Semantic Graph Construction (NetworkX + SentenceTransformer)
   ↓
Phase 3: Keyword Clustering (Node2Vec + K-Means) + Direct Clustering Comparison
   ↓
Phase 4: LLM-based Inconsistency Detection (Gemini 2.5 Pro)
```

## Phases

### Phase 1 — Keyword Extraction
- Two backends: **KeyBERT** (`BAAI/bge-m3`) or **Gemini 2.5 Flash**
- Gemini mode skips KeyBERT normalization in Phase 2
- Output: `results/{dataset}/extraction.json`

### Phase 2 — Semantic Graph
- Encodes keywords with `SentenceTransformer`, adds edges above `semantic_threshold`
- Optional node reduction: satellite nodes absorbed into highest-degree neighbor
- Output: `results/{dataset}/keyword_graph.gexf` (Gephi-compatible)

### Phase 3 — Clustering & Comparison
- **Graph clustering**: Node2Vec embeddings + SVD requirement membership → K-Means; optimal K via elbow method
- **Direct clustering**: SentenceTransformer embeddings of raw requirement sentences → K-Means
- Comparison via ARI + NMI (label-permutation invariant)
- Output: `results/{dataset}/clustering_report.json`, `direct_clustering_report.json`, `clustering_comparison.json`

### Phase 4 — LLM Analysis
- Reads cluster → requirements mapping, sends to **Gemini 2.5 Pro**
- Detects: semantic duplicates, logical inconsistencies, ambiguities, gaps, dependencies
- `--phase4-preview` writes input to file without calling the API
- Output: `results/{dataset}/phase4_analysis.json`

## Installation

1. Clone and enter the project:
```bash
cd Keyword-Graph
```

2. Create and activate virtual environment:
```bash
python -m venv keyword-graph
# Linux/macOS:
source keyword-graph/bin/activate
# Windows:
keyword-graph\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` at repo root with your Gemini API key:
```
MODEL_API_KEY=your-gemini-api-key
```

## Configuration

Dataset-specific configs live in `config/`. Each file sets all paths and parameters for one dataset.

| File | Dataset |
|------|---------|
| `config/test_reale.json` | `data/test_reale.csv` |
| `config/crowd.json` | `data/crowd.csv` |
| `config/run_pipeline.json` | fallback default |

Priority (highest → lowest): **CLI flag** → **config file** → **`DEFAULT_PIPELINE_CONFIG`** in `run_pipeline.py`

## Usage

### Select dataset via flag
```bash
python -m src.run_pipeline --dataset test_reale
python -m src.run_pipeline --dataset crowd
```

### Custom config path
```bash
python -m src.run_pipeline --config config/my_config.json
```

### Full pipeline including Phase 4
```bash
python -m src.run_pipeline --dataset test_reale --with-phase4
```

### Gemini extractor in Phase 1
```bash
python -m src.run_pipeline --dataset test_reale --keyword-extractor gemini
```

### Phase 4 only (phases 1–3 already computed)
```bash
python -m src.run_pipeline --dataset test_reale --phase4-only
```

### Preview Phase 4 input without calling the API
```bash
python -m src.run_pipeline --dataset test_reale --phase4-only --phase4-preview
```

### Use direct clustering report for Phase 4
```bash
python -m src.run_pipeline --dataset test_reale --phase4-only \
  --phase4-use-direct-clustering
```

## Output Files

| File | Phase | Description |
|------|-------|-------------|
| `results/{dataset}/extraction.json` | 1 | Extracted keywords per requirement |
| `results/{dataset}/keyword_graph.gexf` | 2 | Semantic keyword graph |
| `results/{dataset}/clustering_report.json` | 3 | Graph-based clustering results |
| `results/{dataset}/direct_clustering_report.json` | 3 | Direct requirement clustering |
| `results/{dataset}/clustering_comparison.json` | 3 | ARI + NMI comparison |
| `results/{dataset}/phase4_analysis.json` | 4 | LLM inconsistency analysis |

## Project Structure

```
Keyword-Graph/
├── .env                              # MODEL_API_KEY (not committed)
├── requirements.txt
├── config/
│   ├── run_pipeline.json             # Fallback default config
│   ├── test_reale.json               # Config for test_reale dataset
│   └── crowd.json                    # Config for crowd dataset
├── data/
│   ├── test_reale.csv
│   └── crowd.csv
├── results/                          # Pipeline outputs (per dataset)
└── src/
    ├── run_pipeline.py               # Main orchestrator
    ├── phase1/
    │   ├── gemini/
    │   │   ├── extractor.py
    │   │   └── prompt/system_prompt.txt
    │   └── keyBERT/
    │       ├── keybert_adapter.py
    │       ├── normalize.py
    │       ├── prediction.py
    │       └── prepare_data.py
    ├── phase2/
    │   └── graph/
    │       └── build_graph.py
    ├── phase3/
    │   ├── clustering/
    │   │   └── clustering.py
    │   └── comparison/
    │       ├── compare_pipelines.py
    │       └── clustering_comparison_explorer.html
    └── phase4/
        ├── analysis/
        │   └── visualizer.html
        └── llm/
            ├── analyzer.py
            └── prompt/system_prompt.txt
```
