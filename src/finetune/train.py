from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from src.finetune.metrics import keyword_f1, mean_keyword_f1
from src.keywords.normalize import normalize_keyword_text, parse_keyword_text
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

MODEL_NAME = "google/flan-t5-base"
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
}


def _compute_test_keyword_f1(
    trainer: Seq2SeqTrainer,
    tokenized_test,
    tokenizer: T5Tokenizer,
) -> float:
    predictions_output = trainer.predict(tokenized_test)

    pred_ids = predictions_output.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids = predictions_output.label_ids
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return mean_keyword_f1(pred_texts, label_texts)


def build_tokenizer_and_model() -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.tie_word_embeddings = False
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, config=model_config)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    return tokenizer, model


def load_inference_model(model_dir: str | None = None, base_model_name: str = MODEL_NAME) -> tuple[T5Tokenizer, Any]:
    model_path = Path(model_dir) if model_dir else None

    if model_path and model_path.exists():
        tokenizer = T5Tokenizer.from_pretrained(str(model_path))

        if (model_path / "adapter_config.json").exists():
            # Keep original base-model config to avoid embedding mismatch warnings.
            base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            model = T5ForConditionalGeneration.from_pretrained(str(model_path))
    else:
        tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        model = T5ForConditionalGeneration.from_pretrained(base_model_name)

    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    return tokenizer, model


def predict_keywords(
    model_dir: str | None,
    inputs: list[str],
    batch_size: int = 16,
    max_input_len: int = 256,
    max_target_len: int = 64,
    base_model_name: str = MODEL_NAME,
) -> tuple[list[str], dict[str, float | int]]:
    tokenizer, model = load_inference_model(model_dir=model_dir, base_model_name=base_model_name)
    predictions: list[str] = []
    model_generated_count = 0
    fallback_generated_count = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx in range(0, len(inputs), batch_size):
        batch_inputs = inputs[idx : idx + batch_size]
        encoded = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_target_len,
                min_new_tokens=4,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        batch_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for prompt, pred in zip(batch_inputs, batch_preds):
            normalized = normalize_keyword_text(pred)
            if normalized:
                predictions.append(normalized)
                model_generated_count += 1
                continue

            # Deterministic fallback for base-model empty generations.
            predictions.append(_fallback_keywords_from_prompt(prompt, max_keywords=8))
            fallback_generated_count += 1

    total_predictions = len(predictions)
    fallback_ratio = (
        (fallback_generated_count / total_predictions) * 100.0
        if total_predictions
        else 0.0
    )
    print(
        "[diagnostic] keyword_inference "
        f"total={total_predictions} | "
        f"from_model={model_generated_count} | "
        f"from_fallback={fallback_generated_count} | "
        f"fallback_ratio={fallback_ratio:.2f}%"
    )

    diagnostics = {
        "total": total_predictions,
        "from_model": model_generated_count,
        "from_fallback": fallback_generated_count,
        "fallback_ratio": fallback_ratio,
    }

    return predictions, diagnostics


def _fallback_keywords_from_prompt(prompt: str, max_keywords: int = 8) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", str(prompt).lower())
    filtered = [w for w in words if len(w) > 2 and w not in _STOPWORDS]

    # Preserve first occurrence order, then normalize and sort as requested by pipeline rules.
    ordered_unique = list(dict.fromkeys(filtered))
    if not ordered_unique:
        relaxed = [w for w in words if len(w) > 1]
        ordered_unique = list(dict.fromkeys(relaxed))
    if not ordered_unique:
        ordered_unique = ["keyword"]

    heuristic = ", ".join(ordered_unique[:max_keywords])
    return normalize_keyword_text(heuristic)


def compute_keyword_comparison_metrics(predictions: list[str], references: list[str]) -> dict:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_match_count = 0
    sample_f1_scores: list[float] = []

    for pred, ref in zip(predictions, references):
        pred_set = set(parse_keyword_text(pred))
        ref_set = set(parse_keyword_text(ref))

        if pred_set == ref_set:
            exact_match_count += 1

        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        sample_f1_scores.append(keyword_f1(pred, ref))

    precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    total = len(predictions)
    return {
        "samples": total,
        "exact_match": exact_match_count / total if total else 0.0,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "mean_sample_f1": float(sum(sample_f1_scores) / total) if total else 0.0,
    }


def preprocess_fn(batch: dict, tokenizer: T5Tokenizer, max_input_len: int, max_target_len: int) -> dict:
    model_inputs = tokenizer(
        batch["input"],
        max_length=max_input_len,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_target_len,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(
    dataset_dir: str = "data/hf_dataset",
    output_dir: str = "models/flan-t5-keywords",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_input_len: int = 256,
    max_target_len: int = 64,
    use_fp16: bool | None = None,
    require_cuda: bool = False,
) -> dict:
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "logs")

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    device_count = torch.cuda.device_count() if cuda_available else 0
    print(
        f"[device] cuda_available={cuda_available} | torch_cuda_version={cuda_version} | gpu_count={device_count}"
    )

    if cuda_available:
        print(f"[device] gpu_name={torch.cuda.get_device_name(0)}")

    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required but not available. "
            "Your current PyTorch build appears CPU-only (torch.version.cuda is None). "
            "On Windows, activate your project venv and install CUDA wheels, e.g.: "
            "pip uninstall -y torch torchvision torchaudio && "
            "pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio. "
            "Also verify NVIDIA drivers with nvidia-smi."
        )

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found in {dataset_path}. Run prepare_data.py with --save-hf first."
        )

    ds = load_from_disk(str(dataset_path))
    tokenizer, model = build_tokenizer_and_model()

    tokenized = ds.map(
        lambda batch: preprocess_fn(batch, tokenizer, max_input_len, max_target_len),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    resolved_fp16 = False if use_fp16 is None else use_fp16

    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        load_best_model_at_end=True,
        fp16=resolved_fp16,
        max_grad_norm=1.0,
        report_to="none",
    )

    print(f"[device] trainer_device={train_args.device} | fp16={train_args.fp16}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_metrics = trainer.evaluate()
    serializable_metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in eval_metrics.items()
    }

    test_keyword_f1 = _compute_test_keyword_f1(
        trainer=trainer,
        tokenized_test=tokenized["test"],
        tokenizer=tokenizer,
    )
    serializable_metrics["test_keyword_f1"] = float(test_keyword_f1)

    return serializable_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune flan-t5-base for keyword extraction")
    parser.add_argument("--dataset-dir", type=str, default="data/hf_dataset")
    parser.add_argument("--output-dir", type=str, default="models/flan-t5-keywords")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=64)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 training (requires CUDA)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 training",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not available",
    )
    args = parser.parse_args()

    if args.fp16 and args.no_fp16:
        raise ValueError("Use only one of --fp16 or --no-fp16")

    requested_fp16 = True if args.fp16 else False if args.no_fp16 else None

    metrics = train_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        use_fp16=requested_fp16,
        require_cuda=args.require_cuda,
    )
    print(metrics)


if __name__ == "__main__":
    main()
