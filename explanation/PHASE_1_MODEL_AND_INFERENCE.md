# Phase 1: Model and Inference

This step turns user stories into clean keyword sets that the next phases can use.

## Quick Purpose

Phase 1 answers two questions:
- What keywords describe each requirement?
- How reliable is the extraction process?

## What This Step Works On

Primary input:
- `data/crowd.csv`

Expected columns:
- `role`
- `feature`
- `benefit`
- `source_id` (or another requirement identifier)

Optional column:
- `tags` (used only for evaluation metrics, not required for inference)

Optional training input override:
- `--train-input-csv`

## What This Step Produces

- One normalized keyword list per requirement (in memory, passed to Phase 2)
- Optional fine-tuned model artifacts in `models/flan-t5-keywords` (or custom `--model-dir`)
- Verification report in `results/real_case_comparison.json` when `tags` is available
- Inference diagnostics (how often model output vs fallback output was used)

## How This Step Works

### 1. Model selection

Two runtime modes are supported:

1. Training enabled (`--train`)
- Build dataset splits
- Fine-tune FLAN-T5 with LoRA
- Save model to `--model-dir`
- Use this model for inference

2. Training disabled (default)
- Try loading a fine-tuned model from `--model-dir`
- If missing, automatically fallback to base model (`google/flan-t5-base` by default)

### 2. Prompt creation

Each requirement is converted to a prompt using this story template:
- As a <role>, I want <feature> so that <benefit>

The instruction asks for comma-separated keywords only and explicitly discourages generic, low-information terms.

### 3. Generation and fallback

Inference runs in batches and uses beam search.

If generation is empty for a row:
- A deterministic fallback extracts candidate terms from the prompt text
- The output still follows the same normalization pipeline

This ensures each requirement gets a non-empty keyword result.

### 4. Normalization and filtering

All predicted keywords are normalized with consistent rules:
- Lowercasing
- Light singularization
- Deduplication
- Lexicographic sorting
- Stopword and non-informative term filtering

The same rules are applied for:
- Predicted output
- Evaluation metrics
- Graph input parsing in Phase 2

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

[Phase 2 - Graph and Traceability](explanation/PHASE_2_GRAPH_AND_TRACEABILITY.md)
