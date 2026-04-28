# Phase 1: Keyword Extraction

## Executive Summary

Phase 1 automatically identifies and extracts the key concepts from each requirement. This step serves as the foundation for the entire pipeline, converting unstructured requirement text into clean, normalized keyword sets that capture the essential ideas.

## What Phase 1 Achieves

**Input**: Requirements in CSV format with role, feature, and benefit information  
**Output**: Normalized keyword lists for each requirement (passed to Phase 2)

The phase answers two critical questions:
- Which concepts appear most important in each requirement?
- How accurately can we extract keywords from requirement text?

## Input Data Format

The pipeline expects a CSV file (typically `data/crowd.csv`) with these columns:

| Column | Purpose | Required |
|--------|---------|----------|
| `role` | Who is the user (e.g., "homeowner", "administrator") | Yes |
| `feature` | What capability is desired | Yes |
| `benefit` | Why this feature matters | Yes |
| `tags` | Ground-truth keywords (for evaluation only) | No |
| `id` or row index | Unique requirement identifier | Yes |

**Format example**:
```
role: homeowner
feature: control my lighting remotely
benefit: manage energy consumption more efficiently
tags: lighting, remote control, energy management
```

## How It Works

### Step 1: Choose an Extraction Method

The pipeline supports two modes:

**Mode A: KeyBERT (default)**
- Fast, lightweight extraction using pre-trained embeddings
- No training required
- Works immediately out-of-the-box
- Quality depends on source data and keyword definitions
- Automatically used unless otherwise specified

**Mode B: Fine-tuned FLAN-T5**
- Trains a specialized model on provided examples
- Better accuracy when ground-truth keywords (`tags` column) are available
- Higher computational cost
- Produces a reusable model artifact for future use
- Activated with the `--train` flag

### Step 2: Prepare Requirements as Prompts

Each requirement is converted into a structured prompt that guides the extraction model:

```
Task: Extract high-quality keywords from the following user story.

Rules:
- Return ONLY a comma-separated list of keywords.
- Use 1–3 words per keyword (noun phrases preferred).
- Avoid generic or uninformative terms (user, system, home, feature).
- Avoid duplicates and synonyms.
- Prefer domain-specific and meaningful terms.
- Maximum 6 keywords.

Input:
As a homeowner, I want to control my lighting remotely so that I can manage energy consumption more efficiently

Output:
```

The prompt is designed to:
- Encourage concise, meaningful terms
- Discourage common stopwords and generic words
- Limit output to a manageable number
- Request consistent formatting (comma-separated)

### Step 3: Generate Keywords

The extraction model processes each requirement and produces keywords. If the model produces ambiguous or empty output:
- A fallback mechanism extracts candidate terms directly from the requirement text
- Common stopwords and generic terms are filtered out
- The result is normalized to match the standard format
- This ensures every requirement gets a non-empty keyword set

### Step 4: Normalize Keywords

All keywords undergo consistent normalization:

| Operation | Example |
|-----------|---------|
| **Lowercase** | "Lighting" → "lighting" |
| **Remove punctuation** | "lighting's" → "lighting s" → "lighting" |
| **Light singularization** | "controllers" → "controller" |
| **Deduplicate** | Keep only unique terms |
| **Sort alphabetically** | Ensures consistency |

**Result**: A clean, standardized keyword list like `["energy management", "lighting", "remote control"]`

## Output Format

For each requirement, Phase 1 produces:
- A **normalized keyword list** (e.g., `["climate control", "heating", "temperature"]`)
- A **source ID** linking back to the original requirement
- Optional **evaluation metrics** when ground-truth tags are provided

These outputs are held in memory and passed directly to Phase 2 for graph construction.

## Optional: Fine-Tuning Configuration

When `--train` is specified, Phase 1 also creates:
- `models/flan-t5-keywords/` — Fine-tuned model weights
- A dataset split into train/validation/test/real-case subsets
- Evaluation metrics comparing extracted keywords against ground-truth

Training parameters are configurable:
- Epochs, batch size, learning rate
- Training/validation/test split proportions
- Model optimization (LoRA, FP16)

## Quality Indicators

Phase 1 reports diagnostic information:
- Number of requirements processed
- Extraction success rate (model-based vs. fallback)
- When comparison data is available, F1 scores showing extraction accuracy

These metrics help validate whether the extraction quality is sufficient for downstream clustering.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Maximum 6 keywords | Prevents keyword inflation; focuses on primary concepts |
| Stopword filtering | Removes noise from generic terms like "user" or "system" |
| Mandatory fallback | Ensures completeness even with model failures |
| Normalization pipeline | Creates consistency for downstream graph building |
| Requirement traceability | Every keyword can be traced back to source requirements |

## Typical Outcomes

Phase 1 typically produces:
- 80–95% non-empty keyword lists (depending on fallback effectiveness)
- 3–6 keywords per requirement
- High consistency when evaluated against ground-truth (if available)
- Deduplication
- Lexicographic sorting

The same parsing rule is applied for:
- Predicted output
- Evaluation metrics
- Graph input parsing in Phase 2

Stopword filtering happens only inside the deterministic fallback path, not in the shared keyword parser.

### 5. Verification report

If `tags` exists in `data/crowd.csv`, Phase 1 writes:
- `results/real_case_comparison.json`

Main sections include:
- `metrics` (exact match, precision/recall/F1 variants)
- `diagnostics.keyword_inference` (model vs fallback usage)
- `records` (row-level predicted vs reference tags)

If `tags` is missing, inference still runs and reporting is skipped gracefully.

## Why This Step Is Important

Phase 1 creates the semantic foundation of the pipeline:
- Better keyword quality improves graph quality in Phase 2
- Strong normalization keeps behavior consistent across all downstream steps
- Diagnostics make extraction behavior transparent and auditable

## Main Code Entry Points

- `src/run_pipeline.py`
- `src/finetune/prepare_data.py`
- `src/finetune/train.py`
- `src/finetune/metrics.py`
- `src/keywords/normalize.py`

## Next Step

- [Phase 2 - Graph and Traceability](PHASE_2_GRAPH_AND_TRACEABILITY.md)
