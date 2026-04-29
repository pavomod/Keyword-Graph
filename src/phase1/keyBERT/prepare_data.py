from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


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

def prepare_inference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"role", "feature"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    source_ids = df["id"].astype(str) if "id" in df.columns else df.index.astype(str)
    
    return pd.DataFrame(
        {
            "source_id": source_ids,
            "feature": df["feature"].astype(str),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare inference prompts from requirements CSV")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV")
    parser.add_argument(
        "--out-dir", type=str, default="data/processed", help="Directory where output CSV is written"
    )
    parser.add_argument("--sep", type=str, default=",", help="CSV separator (default: ',')")
    args = parser.parse_args()

    input_path = _pick_input_csv(args.input)
    raw_df = pd.read_csv(input_path, sep=args.sep)

    inference_df = prepare_inference_dataframe(raw_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inference_df.to_csv(out_dir / "inference.csv", index=False)

    print(f"Input: {input_path}")
    print(f"Samples prepared: {len(inference_df)}")
    print(f"Output saved in: {out_dir / 'inference.csv'}")


if __name__ == "__main__":
    main()