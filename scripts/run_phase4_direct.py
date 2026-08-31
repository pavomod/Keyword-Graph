"""Run Phase 4 LLM analysis on the DIRECT-clustering baseline partition.

Mirrors run_pipeline.py --phase4-only --phase4-use-direct-clustering,
writing to a separate artifact so the graph-based analysis is preserved.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from src.phase4.llm.analyzer import run_analysis

DATASETS = ["crowd", "test_reale"]
DRY = "--dry" in sys.argv


def main() -> None:
    for dataset in DATASETS:
        cfg = json.loads(Path(f"config/{dataset}.json").read_text(encoding="utf-8"))
        out_name = "phase4_preview_direct.json" if DRY else "phase4_analysis_direct.json"
        output_path = f"results/{dataset}/{out_name}"

        print("=" * 60)
        print(f"[{dataset}] Phase 4 on DIRECT clustering -> {output_path}")
        print("=" * 60)

        run_analysis(
            clustering_report_path=cfg["direct_clustering_report_out"],
            extraction_path=cfg["extraction_out"],
            system_prompt_path=cfg["phase4_system_prompt"],
            output_path=output_path,
            gemini_model=cfg["phase4_gemini_model"],
            gemini_temperature=cfg["phase4_gemini_temperature"],
            gemini_top_p=cfg["phase4_gemini_top_p"],
            gemini_max_tokens=cfg["phase4_gemini_max_tokens"],
            dry_run=DRY,
        )


if __name__ == "__main__":
    main()
