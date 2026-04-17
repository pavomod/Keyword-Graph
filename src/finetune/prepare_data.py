from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def _pick_input_csv(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path

    candidates = [Path("data/crowd.csv"), Path("requirements.csv")]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No input CSV found. Expected data/crowd.csv or requirements.csv"
    )


def _clean_tags(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_prompt(row: pd.Series) -> str:
    role = str(row.get("role", "user")).strip()
    feature = str(row.get("feature", "")).strip()
    benefit = str(row.get("benefit", "")).strip()
    return (
        "Extract comma-separated keywords from this user story. "
        "Return only keywords.\n"
        f"As a {role}, I want {feature} so that {benefit}"
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"role", "feature", "benefit", "tags"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    work_df = df.copy()
    work_df["tags"] = work_df["tags"].apply(_clean_tags)
    work_df = work_df[work_df["tags"] != ""].reset_index(drop=True)

    source_ids = work_df["id"].astype(str) if "id" in work_df.columns else work_df.index.astype(str)
    formatted = pd.DataFrame(
        {
            "source_id": source_ids,
            "input": work_df.apply(_build_prompt, axis=1),
            "target": work_df["tags"],
        }
    )
    return formatted


def split_dataset(df_formatted: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df_formatted, test_size=0.2, random_state=seed, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, shuffle=True
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_dataset_fixed(
    df_formatted: pd.DataFrame,
    train_size: int = 2000,
    validation_size: int = 250,
    test_size: int = 250,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minimum_required = train_size + validation_size + test_size
    if len(df_formatted) < minimum_required:
        raise ValueError(
            "Not enough rows to apply fixed split: "
            f"required={minimum_required}, available={len(df_formatted)}"
        )

    shuffled = df_formatted.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = train_size
    val_end = train_end + validation_size
    test_end = val_end + test_size

    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    val_df = shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test_df = shuffled.iloc[val_end:test_end].reset_index(drop=True)
    real_df = shuffled.iloc[test_end:].reset_index(drop=True)
    return train_df, val_df, test_df, real_df


def to_hf_dataset_dict(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for keyword extraction fine-tuning")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV")
    parser.add_argument(
        "--out-dir", type=str, default="data/processed", help="Directory where split CSV files are written"
    )
    parser.add_argument(
        "--save-hf", action="store_true", help="Also save HuggingFace DatasetDict to disk"
    )
    parser.add_argument(
        "--hf-out-dir",
        type=str,
        default="data/hf_dataset",
        help="Directory where HuggingFace dataset is saved",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--validation-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=250)
    args = parser.parse_args()

    input_path = _pick_input_csv(args.input)
    raw_df = pd.read_csv(input_path)

    formatted_df = prepare_dataframe(raw_df)
    train_df, val_df, test_df, real_df = split_dataset_fixed(
        formatted_df,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "validation.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    real_df.to_csv(out_dir / "real_case.csv", index=False)

    print(f"Input: {input_path}")
    print(f"Samples after cleaning: {len(formatted_df)}")
    print(
        f"Train/Val/Test/Real: {len(train_df)}/{len(val_df)}/{len(test_df)}/{len(real_df)}"
    )
    print(f"CSV splits saved in: {out_dir}")

    if args.save_hf:
        hf_ds = to_hf_dataset_dict(train_df, val_df, test_df)
        hf_out = Path(args.hf_out_dir)
        hf_ds.save_to_disk(str(hf_out))
        print(f"HuggingFace dataset saved in: {hf_out}")


if __name__ == "__main__":
    main()
