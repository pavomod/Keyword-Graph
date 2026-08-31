"""Run the Phase 4 LLM analysis on the DIRECT-clustering baseline partition.

This reproduces the task-based comparison reported in the thesis: the same model,
prompt and parameters used for the graph-based clusters are applied to the clusters
produced by the direct clustering baseline, so that the findings of the two runs can
be compared.

Results are written to `phase4_analysis_direct.json`, leaving the graph-based
`phase4_analysis.json` untouched.

Usage (from anywhere):
    python scripts/run_phase4_direct.py --dry   # preview the model input, no API call
    python scripts/run_phase4_direct.py         # run the analysis (needs MODEL_API_KEY)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Make the project root importable and use it as the base for all relative paths,
# so the script behaves the same regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.phase4.llm.analyzer import run_analysis

DATASETS = ["crowd", "test_reale"]


def main() -> None:
    dry_run = "--dry" in sys.argv

    for dataset in DATASETS:
        config = json.loads((ROOT / "config" / f"{dataset}.json").read_text(encoding="utf-8"))
        out_name = "phase4_preview_direct.json" if dry_run else "phase4_analysis_direct.json"
        output_path = ROOT / "results" / dataset / out_name

        print("=" * 60)
        print(f"[{dataset}] Phase 4 on DIRECT clustering -> {output_path.relative_to(ROOT)}")
        print("=" * 60)

        run_analysis(
            clustering_report_path=str(ROOT / config["direct_clustering_report_out"]),
            extraction_path=str(ROOT / config["extraction_out"]),
            system_prompt_path=str(ROOT / config["phase4_system_prompt"]),
            output_path=str(output_path),
            gemini_model=config["phase4_gemini_model"],
            gemini_temperature=config["phase4_gemini_temperature"],
            gemini_top_p=config["phase4_gemini_top_p"],
            gemini_max_tokens=config["phase4_gemini_max_tokens"],
            dry_run=dry_run,
        )


if __name__ == "__main__":
    main()
