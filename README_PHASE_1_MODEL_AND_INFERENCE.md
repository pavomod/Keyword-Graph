# Phase 1 Technical README: Model, Data, and Inference

This document explains how Phase 1 is implemented in the current pipeline.

## Scope of Phase 1

Phase 1 covers:
- Optional fine-tuning
- Dataset preparation for training
- Full-dataset inference on requirements
- Keyword normalization
- Verification report generation
- Inference diagnostics

Main orchestrator:
- `python -m src.run_pipeline`

Main modules:
- `src/run_pipeline.py`
- `src/finetune/prepare_data.py`
- `src/finetune/train.py`
- `src/finetune/metrics.py`
- `src/keywords/normalize.py`

## Runtime Behavior

The pipeline has two execution modes for model usage:

1. Training enabled (`--train`)
- Prepares fixed train/validation/test splits from the selected training CSV
- Fine-tunes model with LoRA
- Saves model to `--model-dir`
- Uses that model for inference

2. Training disabled (default)
- Tries loading a fine-tuned model from `--model-dir`
- If not available, falls back to base model (`--base-model-name`, default `google/flan-t5-base`)
- Continues inference without stopping

## Data Flow

### Training dataset path

Training CSV is configurable with:
- `--train-input-csv`

If omitted, input fallback logic checks:
- `data/crowd.csv`
- `requirements.csv`

### Inference dataset path

Inference always uses:
- `data/crowd.csv`

This is intentional so graph generation and reporting are computed over the full requirements set.

## Training Dataset Preparation

Training preparation performs:
- Prompt construction from `role`, `feature`, `benefit`
- Target extraction from `tags`
- Cleaning of empty tags
- Fixed split sizes:
  - train: 2000
  - validation: 250
  - test: 250
  - remainder: held-out (not used by trainer)

HF dataset is written to:
- `--hf-dataset-dir` (default `data/hf_dataset`)

## Prompt Template

Each requirement is transformed into a generation prompt:

- Instruction prefix: "Extract comma-separated keywords from this user story. Return only keywords."
- Story body: "As a <role>, I want <feature> so that <benefit>"

## Inference and Generation

Inference runs in batches (`--inference-batch-size`) and uses beam search.

If the model returns an empty prediction for a row:
- A deterministic fallback extracts candidate terms from the prompt text
- Output still goes through the same normalization pipeline

This guarantees non-empty keyword outputs.

## Keyword Normalization Rules

All generated keyword strings are normalized before any metric or graph step.

Normalization guarantees:
- Lowercase
- Singularized words (heuristic rules)
- Deduplicated terms
- Lexicographically sorted output

The same normalization is used in:
- Prediction output
- Evaluation metrics
- Graph input parsing

## Verification Report

If `tags` column exists in `data/crowd.csv`, the pipeline writes:
- `results/real_case_comparison.json`

Report contains:
- `metrics`:
  - exact_match
  - precision_micro
  - recall_micro
  - f1_micro
  - mean_sample_f1
- `diagnostics.keyword_inference`:
  - total
  - from_model
  - from_fallback
  - fallback_ratio
- `records`: per requirement row
  - source_id
  - input
  - predicted_tags
  - real_tags

If `tags` is missing, metrics/report section is skipped.

## Important CLI Controls

Model and training:
- `--train`
- `--no-train`
- `--model-dir`
- `--base-model-name`

Data and splits:
- `--train-input-csv`
- `--train-size`
- `--validation-size`
- `--test-size`

Generation behavior:
- `--max-input-len`
- `--max-target-len`
- `--inference-batch-size`

## Failure and Recovery Strategy

Built-in resilience in Phase 1:
- Missing fine-tuned model does not fail pipeline; it falls back to base model
- Empty generations are recovered through prompt-based fallback extraction
- Diagnostic counters expose how often fallback was used

## Outputs Produced by Phase 1

- Trained model artifacts (when `--train`): `models/flan-t5-keywords` (or custom path)
- Verification JSON: `results/real_case_comparison.json`
- In-memory normalized keyword lists passed to Phase 2
