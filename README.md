# Keyword-Graph: Requirement Analysis Pipeline

A comprehensive pipeline for analyzing software requirements through keyword extraction, graph construction, clustering, and LLM-based inconsistency detection.

## Pipeline Overview

```
Phase 1: Keyword Extraction from Requirements (KeyBERT or Gemini 2.5 Pro)
   ↓
Phase 2: Semantic Graph Construction (NetworkX)
   ↓
Phase 3: Keyword Clustering and Visualization (Node2Vec + K-Means)
   ↓
Phase 4: LLM-based Inconsistency Detection (Gemini 2.5 Pro)
```

## Phases Description

### Phase 1: Keyword Extraction
- Extracts important keywords and keyphrases from requirements
- Uses **KeyBERT** with BAAI/bge-m3 model or **Gemini 2.5 Flash-Lite**
- Gemini mode sends the requirement text directly without the KeyBERT-style cleanup step
- Produces: `results/extraction.json`

### Phase 2: Semantic Graph Construction
- Builds a semantic similarity graph of keywords
- Uses **sentence-transformers** for semantic similarity
- Reduces graph complexity through node pruning
- Produces: `results/keyword_graph.gexf`

### Phase 3: Clustering & Visualization
- Clusters keywords using **Node2Vec embeddings** + **K-Means**
- Compares two clustering approaches (keyword-graph vs direct)
- Provides elbow method analysis for optimal k selection
- Produces: `results/clustering_report.json`, `results/direct_clustering_report.json`

### Phase 4: LLM-based Analysis
- Analyzes requirement clusters with **Gemini 2.5 Pro**
- Detects: duplicates, inconsistencies, ambiguities, gaps, dependencies
- Produces: `results/phase4_analysis.json`

## Installation

### Setup

1. Clone and navigate to project:
```bash
cd Keyword-Graph
```

2. Create and activate virtual environment:
```bash
python -m venv keyword-graph
source keyword-graph/bin/activate  # On Windows: keyword-graph\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API key:
```bash
export MODEL_API_KEY="your-gemini-api-key"  # On Windows: set MODEL_API_KEY=...
```

## Configuration

Main configuration: `config/run_pipeline.json`

## Usage

### Pipeline whitout Phase 4

```bash
python -m src.run_pipeline
```

### Full Pipeline Including Phase 4

```bash
python -m src.run_pipeline --with-phase4
```

### Using Gemini in Phase 1

```bash
python -m src.run_pipeline --keyword-extractor gemini
```

You can configure the Gemini model for Phase 1 in `config/run_pipeline.json` with `phase1_gemini_model`, which defaults to `gemini-2.5-flash-lite`.

### Phase 4 Only

Analyze existing clusters without re-running phases 1-3:

```bash
python -m src.run_pipeline --phase4-only
```

### Phase 4 Input Preview

Create the input without call the API

```bash
python -m src.run_pipeline --phase4-only --phase4-preview
```

### Using Direct Clustering for Phase 4

```bash
python -m src.run_pipeline. --phase4-only \
  --phase4-use-direct-clustering \
  --phase4-clustering-report results/direct_clustering_report.json
```

## Output Files

| File | Phase | Description |
|------|-------|-------------|
| `results/requirements_file_name/extraction.json` | 1 | Extracted keywords for each requirement |
| `results/requirements_file_name/keyword_graph.gexf` | 2 | Semantic graph in GEXF format |
| `results/requirements_file_name/clustering_report.json` | 3 | Keyword-graph clustering analysis |
| `results/requirements_file_name/direct_clustering_report.json` | 3 | Direct requirement clustering |
| `results/requirements_file_name/clustering_comparison.json` | 3 | Comparison of two approaches |
| `results/requirements_file_name/phase4_analysis.json` | 4 | LLM-based inconsistency analysis |


### What Phase 4 Analyzes

For each cluster, detects:

1. **Semantic Duplicates**: Requirements with identical or overlapping meaning
2. **Logical Inconsistencies**: Contradictions or conflicts between requirements
3. **Ambiguities**: Vague or underspecified requirements
4. **Missing Requirements**: Gaps or implicit dependencies
5. **Dependencies**: How requirements relate to each other

### Output Example

```json
{
  "clusters": [
    {
      "cluster_id": "1",
      "duplicates": [
        {
          "requirements": ["1", "5"],
          "type": "partial",
          "reason": "Both reference user registration points"
        }
      ],
      "inconsistencies": [...],
      "ambiguities": [...],
      "gaps": [...],
      "dependencies": [...]
    }
  ]
}
```


## Project Structure

```
Keyword-Graph/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── config/
│   └── run_pipeline.json             # Pipeline configuration
├── data/                             # Input requirements
├── results/                          # Output files
└── src/
    ├── run_pipeline.py               # Main entry point
    ├── phase1/                       # Keyword extraction
    │   └── keyBERT/
    ├── phase2/                       # Graph construction
    │   └── graph/
    ├── phase3/                       # Clustering
    │   ├── clustering/
    │   └── comparison/
    └── phase4/                       # LLM analysis (NEW)
        └── llm/
            ├── analyzer.py           # Gemini analysis logic
            ├── prompt/
            │   └── system_prompt.txt # LLM system prompt
            └── README.md             # Phase 4 documentation
```