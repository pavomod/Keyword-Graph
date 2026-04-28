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
- Inference diagnostics showing how many rows came from the model vs fallback

## How This Step Works

### 1. Model selection

Two runtime modes are supported:

1. Training enabled (`--train`)
- Build dataset splits
- Fine-tune FLAN-T5 with LoRA
- Save model to `--model-dir`
- Use this model for inference

2. Training disabled (default)
1. Default extractor: KeyBERT (preferred)
- If `keybert` is installed in the environment, the pipeline uses KeyBERT to extract keywords by default when no fine-tuned model is explicitly selected.

2. Fine-tuned model
- If `--use-finetuned` is provided (or if `--train` was run and produced a checkpoint in `--model-dir`), the pipeline loads and uses the fine-tuned FLAN-T5 model.
- If `--use-finetuned` is requested but no checkpoint is present, the pipeline warns and falls back to KeyBERT or the base model.

Note: Install KeyBERT in your environment to enable the default extractor behavior (see `requirements.txt`).

### 2. Prompt creation

Each requirement is converted to a prompt using the full instruction template in `src/finetune/prepare_data.py`.
The prompt asks for comma-separated keywords only, caps the number of keywords, and discourages generic or low-information terms such as `user`, `system`, `home`, and `feature`.

Actual prompt template:

```text
Task: Extract high-quality keywords from the following user story.
Rules:
- Return ONLY a comma-separated list of keywords.
- Use 1–3 words per keyword (noun phrases preferred).
- Avoid generic or uninformative terms (e.g., user, system, home, feature).
- Avoid duplicates and synonyms.
- Prefer domain-specific and meaningful terms.
- Include actions only if they are essential (e.g., 'user authentication').
- Maximum 6 keywords.
Example:
Input: As a user, I want to reset my password so that I can regain access.
Output: password reset, account access, authentication

Input:
As a <role>, I want <feature> so that <benefit>
Output:
```

In code, `<role>`, `<feature>`, and `<benefit>` are replaced with the row values from the CSV.

### 3. Generation and fallback

Inference runs in batches and uses beam search.

If generation normalizes to an empty string for a row:
- A deterministic fallback extracts candidate terms from the prompt text
- The fallback removes common stopwords and generic terms before normalization
- The output still follows the same normalization pipeline

This ensures each requirement gets a non-empty keyword result.

### 4. Normalization and filtering

All predicted keywords are normalized with consistent rules:
- Lowercasing
- Light singularization
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
