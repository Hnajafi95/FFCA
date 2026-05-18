"""Diff v0.5 vs v0.6 narrations on the same 14 cases.

Reports per case:
  - whether v0.6 added rule-free observations (and how many)
  - executive-summary length and shape changes
  - whether v0.6 actions reference the same rule IDs
  - any case where the headline action's title diverged

Run after narrate_v06.py.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# v0.5 narrations live at known paths
V05_LOCATIONS = {
    # engineered v0.5 cases
    "v05/credit_loan":                ("credit_loan", None),
    "v05/california_housing_leak":    ("california_housing_leak", None),
    "v05/california_housing_spurious":("california_housing_spurious", None),
    "v05/bike_sharing":               ("bike_sharing", None),
    "v05/wine_quality":               ("wine_quality", None),
    "v05/waterbirds":                 ("waterbirds", None),
    # flooding gate
    "flooding/gate/before_3hr":  ("flooding_narrations/gate", "before_3hr"),
    "flooding/gate/after_3hr":   ("flooding_narrations/gate", "after_3hr"),
    "flooding/gate/before_6hr":  ("flooding_narrations/gate", "before_6hr"),
    "flooding/gate/after_6hr":   ("flooding_narrations/gate", "after_6hr"),
    "flooding/gate/before_12hr": ("flooding_narrations/gate", "before_12hr"),
    "flooding/gate/after_12hr":  ("flooding_narrations/gate", "after_12hr"),
    "flooding/gate/before_24hr": ("flooding_narrations/gate", "before_24hr"),
    "flooding/gate/after_24hr":  ("flooding_narrations/gate", "after_24hr"),
}


def _read_v05_findings(label: str) -> dict | None:
    base, sub = V05_LOCATIONS[label]
    p = REPO / "FFCA_runs_results_v04_real" / base
    if sub:
        p = p / sub
    f = p / "findings_v05.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _read_v06_findings(label: str) -> dict | None:
    base, sub = V05_LOCATIONS[label]
    p = REPO / "FFCA_runs_results_v04_real" / base
    if sub:
        p = p / sub
    f = p / "findings_v06.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _read_v05_summary(label: str) -> str:
    """v0.5 executive summaries live inside findings_v05.json (flooding) or
    must be parsed out of diagnosis_v5.md (engineered cases — the JSON there
    doesn't carry exec_summary)."""
    base, sub = V05_LOCATIONS[label]
    p = REPO / "FFCA_runs_results_v04_real" / base
    if sub:
        p = p / sub
    fj = p / "findings_v05.json"
    if fj.exists():
        d = json.loads(fj.read_text())
        if "executive_summary" in d:
            return d["executive_summary"]
    md = p / "diagnosis_v5.md"
    if md.exists():
        txt = md.read_text()
        # exec summary lives between "## Executive summary" and the next h2
        marker = "## Executive summary"
        if marker in txt:
            start = txt.index(marker) + len(marker)
            tail = txt[start:].lstrip()
            next_h2 = tail.find("\n## ")
            return (tail if next_h2 < 0 else tail[:next_h2]).strip()
    return ""


def main() -> None:
    rows = []
    for label in V05_LOCATIONS:
        v05 = _read_v05_findings(label)
        v06 = _read_v06_findings(label)
        if not v06:
            print(f"[skip] {label}: v0.6 findings missing")
            continue
        v05_exec = _read_v05_summary(label)
        v06_exec = v06.get("executive_summary", "")

        v05_rules = set(v05.get("diagnostic_rule_ids", [])) if v05 else set()
        v06_rules = set(v06.get("diagnostic_rule_ids", []))
        rules_added = v06_rules - v05_rules
        rules_dropped = v05_rules - v06_rules

        rows.append({
            "label": label,
            "intent": v06.get("intent"),
            "v05_exec_len": len(v05_exec.split()),
            "v06_exec_len": len(v06_exec.split()),
            "n_rule_free_observations_v06": len(v06.get("rule_free_observations", [])),
            "diagnostic_rules_added_v06": sorted(rules_added),
            "diagnostic_rules_dropped_v06": sorted(rules_dropped),
            "rule_free_observations": v06.get("rule_free_observations", []),
        })

    # Print summary table
    print(f"\n{'label':40} {'intent':10} {'v05_words':>9} {'v06_words':>9} {'obs':>4}  {'rule_diff':<30}")
    print("-" * 130)
    for r in rows:
        diff = ""
        if r["diagnostic_rules_added_v06"]:
            diff += "+" + ",".join(r["diagnostic_rules_added_v06"])
        if r["diagnostic_rules_dropped_v06"]:
            if diff: diff += " "
            diff += "-" + ",".join(r["diagnostic_rules_dropped_v06"])
        print(f"{r['label']:40} {r['intent'] or '-':10} {r['v05_exec_len']:>9} {r['v06_exec_len']:>9} "
              f"{r['n_rule_free_observations_v06']:>4}  {diff[:50]}")

    # Detailed observation listing
    print("\n\n=== rule-free observations produced by v0.6 ===\n")
    for r in rows:
        if not r["rule_free_observations"]:
            continue
        print(f"--- {r['label']}  (intent={r['intent']}) ---")
        for o in r["rule_free_observations"]:
            print(f"  • {o.get('what','')}")
            print(f"    evidence: {o.get('evidence','')}")
        print()

    out = REPO / "FFCA_runs_results_v04_real" / "diff_v05_v06.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
