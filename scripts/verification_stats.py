"""Summarise the manual verification of the Phase 4 findings.

Reads the verdicts recorded in scripts/verdicts/ and reports, for the graph-based
partition and for the direct-clustering baseline: the number of confirmed, borderline
and false-positive findings, the false-positive rate per category, and how many of the
false positives are gaps whose supposedly missing requirement exists in another cluster
(artifacts of the per-cluster isolation rule).

Usage (from anywhere):
    python scripts/verification_stats.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["duplicates", "inconsistencies", "ambiguities", "gaps", "dependencies"]

RUNS = [
    ("uw_graph", "Uniwhere", "Graph"),
    ("uw_base", "Uniwhere", "Baseline"),
    ("v2i_graph", "V2I", "Graph"),
    ("v2i_base", "V2I", "Baseline"),
]

# False positives that are gaps whose "missing" requirement exists in another cluster.
# Listed explicitly so the count is auditable rather than inferred from the wording.
ISOLATION_ARTIFACTS = {
    ("Uniwhere", "Graph"): ["C3|gaps|1", "C3|gaps|2", "C4|gaps|2", "C6|gaps|1", "C7|gaps|2"],
    ("V2I", "Graph"): ["C1|gaps|3", "C2|gaps|2", "C7|gaps|1"],
    ("Uniwhere", "Baseline"): ["C12|gaps|1", "C11|gaps|1", "C11|gaps|2", "C9|gaps|1",
                               "C9|gaps|2", "C1|gaps|1", "C4|gaps|1", "C6|gaps|2"],
    ("V2I", "Baseline"): ["C1|gaps|1", "C6|gaps|1", "C0|gaps|1"],
}


def load() -> dict[tuple[str, str], dict[str, str]]:
    runs: dict[tuple[str, str], dict[str, str]] = {}
    for name, dataset, partition in RUNS:
        path = ROOT / "scripts" / "verdicts" / f"{name}.json"
        runs[(dataset, partition)] = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    return runs


def verdict_of(raw: str) -> str:
    return raw.split("|")[0]


def main() -> None:
    runs = load()

    for (dataset, partition), verdicts in runs.items():
        for key in ISOLATION_ARTIFACTS[(dataset, partition)]:
            assert verdict_of(verdicts[key]) == "FP", f"{dataset}/{partition} {key} is not a false positive"

    print("=" * 74)
    print("VERIFICATION OUTCOME")
    print("=" * 74)
    print(f"{'run':<22}{'n':>5}{'confirmed':>11}{'borderline':>12}{'FP':>5}{'FP rate':>9}")
    for name, dataset, partition in RUNS:
        v = runs[(dataset, partition)]
        c = Counter(verdict_of(x) for x in v.values())
        n = len(v)
        print(f"{dataset + ' / ' + partition:<22}{n:>5}{c['TP']:>11}{c['BORDERLINE']:>12}"
              f"{c['FP']:>5}{100 * c['FP'] / n:>8.1f}%")

    print()
    for partition in ("Graph", "Baseline"):
        c = Counter(verdict_of(x) for (d, p), v in runs.items() if p == partition for x in v.values())
        n = sum(c.values())
        print(f"{partition + ' (both datasets)':<22}{n:>5}{c['TP']:>11}{c['BORDERLINE']:>12}"
              f"{c['FP']:>5}{100 * c['FP'] / n:>8.1f}%")

    print()
    print("=" * 74)
    print("FALSE POSITIVES BY CATEGORY")
    print("=" * 74)
    print(f"{'category':<18}{'Graph n':>9}{'FP':>5}{'rate':>8}{'Baseline n':>12}{'FP':>5}{'rate':>8}")
    for category in CATEGORIES:
        cells = []
        for partition in ("Graph", "Baseline"):
            n = fp = 0
            for (d, p), v in runs.items():
                if p != partition:
                    continue
                for key, raw in v.items():
                    if key.split("|")[1] == category:
                        n += 1
                        fp += verdict_of(raw) == "FP"
            cells.append((n, fp))
        (gn, gf), (bn, bf) = cells
        print(f"{category:<18}{gn:>9}{gf:>5}{100 * gf / gn if gn else 0:>7.1f}%"
              f"{bn:>12}{bf:>5}{100 * bf / bn if bn else 0:>7.1f}%")

    print()
    print("=" * 74)
    print("FALSE GAPS CAUSED BY THE ISOLATION RULE")
    print("=" * 74)
    for partition in ("Graph", "Baseline"):
        fp = sum(1 for (d, p), v in runs.items() if p == partition
                 for x in v.values() if verdict_of(x) == "FP")
        artifacts = sum(len(k) for (d, p), k in ISOLATION_ARTIFACTS.items() if p == partition)
        print(f"{partition:<10} false positives={fp:<4} of which isolation artifacts={artifacts}"
              f" ({100 * artifacts / fp:.0f}%)")


if __name__ == "__main__":
    main()
