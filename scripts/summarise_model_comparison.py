"""Summarise the graph / baseline / no-clustering comparison run with one model.

Usage (from the repository root):
    python scripts/summarise_model_comparison.py [model]   # default gemini-3.1-pro-preview
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["duplicates", "inconsistencies", "ambiguities", "gaps", "dependencies"]


def ids_of(item: dict) -> set[str]:
    ids = {str(x) for v in item.values() if isinstance(v, list) for x in v if str(x).isdigit()}
    if str(item.get("requirement", "")).isdigit():
        ids.add(str(item["requirement"]))
    return ids


def summarise(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "clusters" not in data:
        return {"error": data.get("error", "no clusters in output")}
    mapping = {str(k): str(v) for k, v in data["requirement_to_cluster_mapping"].items()}
    counts = {c: 0 for c in CATEGORIES}
    consolidation = outside = 0
    for cluster in data["clusters"]:
        cid = str(cluster.get("cluster_id"))
        for cat in CATEGORIES:
            for item in cluster.get(cat, []):
                counts[cat] += 1
                if any(mapping.get(i) != cid for i in ids_of(item)):
                    outside += 1
                if cat == "duplicates":
                    consolidation += max(0, len(item.get("requirements", [])) - 1)
    scores = [c.get("quality_score") for c in data["clusters"] if isinstance(c.get("quality_score"), (int, float))]
    return counts | {
        "total": sum(counts.values()),
        "consolidation": consolidation,
        "clusters": len(data["clusters"]),
        "mean_coherence": round(sum(scores) / len(scores), 1) if scores else None,
        "outside_cluster": outside,
    }


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.1-pro-preview"
    folder = ROOT / "results" / "model_comparison" / model
    usage = json.loads((ROOT / "results" / "token_runs" / "token_usage.json").read_text(encoding="utf-8"))
    header = ["dataset", "strategy", "clusters", *CATEGORIES, "total", "consolidation", "coherence",
              "outside", "in_tok", "out_tok", "think_tok", "total_tok", "sec"]
    print(" | ".join(header))
    for dataset in ["crowd", "test_reale"]:
        for strategy in ["graph", "baseline", "flat"]:
            path = folder / f"{dataset}_{strategy}.json"
            if not path.exists():
                print(dataset, strategy, "missing")
                continue
            s = summarise(path)
            u = usage.get(f"{model}/{dataset}/{strategy}", {})
            if "error" in s:
                print(dataset, strategy, "ERROR:", s["error"])
                continue
            row = [dataset, strategy, s["clusters"], *[s[c] for c in CATEGORIES], s["total"], s["consolidation"],
                   s["mean_coherence"], s["outside_cluster"], u.get("prompt"), u.get("output"), u.get("thinking"),
                   u.get("total"), u.get("seconds")]
            print(" | ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
