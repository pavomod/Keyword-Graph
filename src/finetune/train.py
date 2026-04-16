from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from src.finetune.metrics import mean_keyword_f1
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

MODEL_NAME = "google/flan-t5-base"


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
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.tie_word_embeddings = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    return tokenizer, model


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
