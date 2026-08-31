"""Compare the Phase 4 findings obtained on the graph-based partition and on the baseline.

For each dataset it prints, side by side, the number of findings per category, the number
of consolidation opportunities implied by the duplicate groups, the coherence scores
assigned by the model, and a check that every cited requirement belongs to the cluster
being analysed.

Requires `phase4_analysis.json` (graph-based) and `phase4_analysis_direct.json`
(baseline, produced by run_phase4_direct.py).

Usage (from anywhere):
    python scripts/compare_phase4.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["duplicates", "inconsistencies", "ambiguities", "gaps", "dependencies"]
DATASETS = [("crowd", "Uniwhere"), ("test_reale", "V2I case")]

METRICS = [
    ("clusters", "Clusters"),
    ("duplicates", "Duplicate groups"),
    ("consolidations", "Consolidation opportunities"),
    ("inconsistencies", "Inconsistencies"),
    ("ambiguities", "Ambiguities"),
    ("gaps", "Gaps"),
    ("dependencies", "Dependencies"),
    ("total", "Total findings"),
    ("cited_ids", "Cited requirement ids"),
    ("violations", "Out-of-cluster citations"),
    ("mean_score", "Mean coherence"),
    ("min_score", "Minimum coherence"),
    ("max_score", "Maximum coherence"),
]


def load(dataset: str, name: str) -> dict:
    return json.loads((ROOT / "results" / dataset / name).read_text(encoding="utf-8"))


def collect_ids(finding: dict) -> set[str]:
    """Every requirement id a finding refers to, across the different field shapes."""
    ids: set[str] = set()
    for key in ("requirements", "related_to"):
        ids.update(str(i) for i in (finding.get(key) or []))
    for key in ("requirement", "from", "to"):
        if finding.get(key):
            ids.add(str(finding[key]))
    return ids


def summarise(analysis: dict, cluster_mapping: dict[str, list[str]]) -> dict:
    clusters = analysis.get("clusters", [])
    counts = {category: 0 for category in CATEGORIES}
    consolidations = cited = violations = 0
    scores: list[float] = []

    for cluster in clusters:
        allowed = {str(r) for r in cluster_mapping.get(str(cluster["cluster_id"]), [])}
        ids: set[str] = set()

        for category in CATEGORIES:
            findings = cluster.get(category) or []
            counts[category] += len(findings)
            for finding in findings:
                if category == "duplicates":
                    # merging a group of n requirements removes n-1 of them
                    consolidations += max(0, len(finding.get("requirements") or []) - 1)
                ids |= collect_ids(finding)

        cited += len(ids)
        violations += len(ids - allowed)

        try:
            scores.append(float(cluster["quality_score"]))
        except (KeyError, TypeError, ValueError):
            pass

    return {
        "clusters": len(clusters),
        **counts,
        "consolidations": consolidations,
        "total": sum(counts.values()),
        "cited_ids": cited,
        "violations": violations,
        "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
    }


def main() -> None:
    for dataset, label in DATASETS:
        graph = summarise(
            load(dataset, "phase4_analysis.json"),
            load(dataset, "clustering_report.json")["cluster_requirements_mapping"],
        )
        baseline = summarise(
            load(dataset, "phase4_analysis_direct.json"),
            load(dataset, "direct_clustering_report.json")["cluster_requirements_mapping"],
        )

        print("=" * 62)
        print(label)
        print("=" * 62)
        print(f"{'':<30}{'Graph':>14}{'Baseline':>14}")
        print("-" * 62)
        for key, caption in METRICS:
            print(f"{caption:<30}{str(graph[key]):>14}{str(baseline[key]):>14}")
        print()


if __name__ == "__main__":
    main()
