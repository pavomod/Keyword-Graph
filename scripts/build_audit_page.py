"""Build a standalone HTML page for reviewing the verified Phase 4 findings.

Reads the worksheet produced by export_findings.py and embeds it into a single
self-contained page: no server, no build step, no network access needed beyond the
web fonts, which fall back cleanly when unavailable. The page can be opened by
double-clicking it and sent as an email attachment.

    results/verification/findings_audit.html

Run export_findings.py first if the verdicts have changed.

Usage (from anywhere):
    python scripts/build_audit_page.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = ROOT / "scripts" / "templates" / "audit_page.html"
FINDINGS = ROOT / "results" / "verification" / "findings.json"
OUTPUT = ROOT / "results" / "verification" / "findings_audit.html"

PLACEHOLDER = "__DATA__"


def main() -> None:
    if not FINDINGS.exists():
        raise SystemExit("findings.json not found - run scripts/export_findings.py first")

    data = FINDINGS.read_text(encoding="utf-8")
    if "</script" in data.lower():
        raise SystemExit("findings.json contains a closing script tag and cannot be inlined safely")

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"template is missing the {PLACEHOLDER} placeholder")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template.replace(PLACEHOLDER, data), encoding="utf-8")

    rows = json.loads(data)
    judged = sum(1 for r in rows if r["verdict"] != "UNJUDGED")
    size_kb = round(OUTPUT.stat().st_size / 1024)
    print(f"wrote {OUTPUT.relative_to(ROOT)} - {len(rows)} findings ({judged} judged), {size_kb} KB")


if __name__ == "__main__":
    main()
