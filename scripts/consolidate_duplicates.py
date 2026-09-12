"""Ask the model to merge a group of duplicate requirements into a single one.

The per-cluster analysis reports which requirements say the same thing; the natural
next step for the analyst is to replace each such group with one requirement. This
script performs that step on a chosen duplicate group and writes the result to
results/verification/consolidation.json.

The prompt asks for a requirement that preserves every constraint present in the
inputs, adds nothing, and states explicitly any point on which the inputs disagree,
so that a merge never silently resolves a decision that belongs to the stakeholders.

Usage (from anywhere):
    python scripts/consolidate_duplicates.py                     # the groups below
    python scripts/consolidate_duplicates.py --dry               # show the prompts only
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """You are a requirements analyst consolidating redundant requirements.

You receive a group of requirements that a previous analysis flagged as expressing the
same need. Produce a single requirement that replaces all of them.

RULES
- Preserve every constraint, actor and condition that appears in any input requirement.
- Add nothing that is not stated in the inputs. Do not invent thresholds or actors.
- Use a single, plainly worded sentence in the imperative form used by the inputs.
- If the inputs disagree on any point, or express the same constraint in units that do
  not coincide, do NOT choose for the stakeholders: report the disagreement separately.

OUTPUT (strict JSON, no text outside it)
{
  "consolidated": "the single replacement requirement",
  "covered": ["id", "id"],
  "decisions_required": ["each point the inputs leave open or disagree on"],
  "dropped": ["any information in the inputs the merged requirement does not carry"]
}"""

# (dataset folder, analysis file, cluster id, index of the duplicate group, label)
GROUPS = [
    ("crowd", "phase4_analysis.json", "8", 1, "Maximum stay duration"),
    ("crowd", "phase4_analysis.json", "1", 1, "Purchasing points"),
]


def requirement_texts(dataset: str) -> dict[str, str]:
    data = json.loads((ROOT / "results" / dataset / "extraction.json").read_text(encoding="utf-8"))
    return {str(r["source_id"]): r["input"].strip() for r in data["records"]}


def duplicate_group(dataset: str, analysis: str, cluster_id: str, index: int) -> dict:
    data = json.loads((ROOT / "results" / dataset / analysis).read_text(encoding="utf-8"))
    cluster = next(c for c in data["clusters"] if str(c["cluster_id"]) == cluster_id)
    return cluster["duplicates"][index - 1]


def build_input(texts: dict[str, str], group: dict) -> str:
    lines = [f"{rid}: {texts[str(rid)]}" for rid in group.get("requirements", [])]
    return "REQUIREMENTS FLAGGED AS DUPLICATES\n\n" + "\n".join(lines)


def main() -> None:
    dry_run = "--dry" in sys.argv
    results = []

    for dataset, analysis, cluster_id, index, label in GROUPS:
        texts = requirement_texts(dataset)
        group = duplicate_group(dataset, analysis, cluster_id, index)
        user_message = build_input(texts, group)

        print("=" * 70)
        print(f"{label}  (dataset {dataset}, cluster C{cluster_id}, group #{index})")
        print("=" * 70)
        print(user_message)
        print()

        if dry_run:
            continue

        import google.generativeai as genai
        import os

        genai.configure(api_key=os.getenv("MODEL_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(
            [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}, {"text": user_message}]}],
            generation_config=genai.types.GenerationConfig(temperature=0.2, top_p=0.9,
                                                           max_output_tokens=12000),
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            raw = raw[4:] if raw.startswith("json") else raw
        parsed = json.loads(raw.strip())

        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print()
        results.append({"dataset": dataset, "cluster": cluster_id, "group": index,
                        "label": label, "inputs": {str(r): texts[str(r)]
                                                    for r in group.get("requirements", [])},
                        "result": parsed})

    if results:
        out = ROOT / "results" / "verification" / "consolidation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
