"""Walk validation_runs/0?_*/ and emit a summary.md per test directory.

For each test, the script pairs `*_baseline` subdirs with `*_with` (or
`*_with_improvements`) subdirs and compares the two FFCA reports:

  * Archetype distribution per checkpoint
  * Diagnostic findings unique to each mode (by name)
  * Interaction-strength range (min / max / mean across the final checkpoint)
  * Top-K features Jaccard overlap

Usage
-----
    python validation_runs/summarize_validation.py
    python validation_runs/summarize_validation.py --top-k 10 --root validation_runs

The output `summary.md` is written into each test directory next to the
paired subdirs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

WITH_SUFFIXES = ("_with_improvements", "_with")
BASE_SUFFIXES = ("_baseline",)
REPORT_FNAME = "report.json"

# Keep in sync with ffca.core.archetypes.ARCHETYPE_NAMES
ARCHETYPE_NAMES = [
    "Noise",
    "Hidden Interactor",
    "Workhorse",
    "Catalyst",
    "Nonlinear Driver",
    "Volatile Specialist",
    "Stable Contributor",
    "Complex Driver",
]


def _archetype_label(idx) -> str:
    try:
        i = int(idx)
        if 0 <= i < len(ARCHETYPE_NAMES):
            return ARCHETYPE_NAMES[i]
    except (TypeError, ValueError):
        pass
    return str(idx)


def _strip_suffix(name: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    for s in suffixes:
        if name == s.lstrip("_"):
            return ""
        if name.endswith(s):
            return name[: -len(s)]
    return None


def _find_pairs(test_dir: pathlib.Path) -> List[Tuple[str, pathlib.Path, pathlib.Path]]:
    """Return list of (stem, baseline_dir, with_dir) for a single test."""
    base_map: Dict[str, pathlib.Path] = {}
    with_map: Dict[str, pathlib.Path] = {}
    for child in sorted(test_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / REPORT_FNAME).exists():
            continue
        stem = _strip_suffix(child.name, BASE_SUFFIXES)
        if stem is not None:
            base_map[stem] = child
            continue
        stem = _strip_suffix(child.name, WITH_SUFFIXES)
        if stem is not None:
            with_map[stem] = child
    pairs: List[Tuple[str, pathlib.Path, pathlib.Path]] = []
    for stem in sorted(set(base_map) | set(with_map)):
        if stem in base_map and stem in with_map:
            pairs.append((stem, base_map[stem], with_map[stem]))
    return pairs


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _archetype_distribution(report: dict, ckpt_idx: int) -> Counter:
    sigs = report.get("signatures", [])
    if not sigs or ckpt_idx >= len(sigs):
        return Counter()
    arche = sigs[ckpt_idx].get("archetypes", []) or []
    return Counter(_archetype_label(a) for a in arche)


def _findings_by_name(report: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for f in report.get("findings", []) or []:
        out[f.get("name", "?")] = f
    return out


def _final_interaction_stats(report: dict) -> Optional[dict]:
    sigs = report.get("signatures", [])
    if not sigs:
        return None
    last = sigs[-1].get("interaction", []) or []
    if not last:
        return None
    last = [float(v) for v in last]
    n = len(last)
    return {
        "n": n,
        "min": min(last),
        "max": max(last),
        "mean": sum(last) / n,
    }


def _top_features(report: dict, k: int) -> List[str]:
    sigs = report.get("signatures", [])
    if not sigs:
        return []
    inter = sigs[-1].get("interaction", []) or []
    names = sigs[-1].get("feature_names") or report.get("feature_names") or []
    if not inter or not names:
        return []
    ranked = sorted(zip(names, inter), key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in ranked[:k]]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _fmt_dist(dist: Counter, total: int) -> str:
    if total == 0 or not dist:
        return "_(none)_"
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"`{k}`={v} ({100*v/total:.0f}%)" for k, v in items)


def _format_pair_section(stem: str,
                         baseline_dir: pathlib.Path,
                         with_dir: pathlib.Path,
                         top_k: int) -> str:
    base = _load(baseline_dir / REPORT_FNAME)
    impr = _load(with_dir / REPORT_FNAME)

    labels = base.get("checkpoint_labels") or impr.get("checkpoint_labels") or []
    lines: List[str] = []
    title = stem if stem else baseline_dir.parent.name
    lines.append(f"## Pair: `{title}`")
    lines.append("")
    lines.append(f"- **Baseline:** `{baseline_dir.name}/`")
    lines.append(f"- **With improvements:** `{with_dir.name}/`")
    n_features = base.get("n_features") or impr.get("n_features") or "?"
    lines.append(f"- **Checkpoints:** {labels}")
    lines.append(f"- **Features:** {n_features}")
    lines.append("")

    # Archetype distribution per checkpoint
    lines.append("### Archetype distribution per checkpoint")
    lines.append("")
    lines.append("| Checkpoint | Baseline | With improvements |")
    lines.append("|---|---|---|")
    n_ckpt = max(len(base.get("signatures", [])), len(impr.get("signatures", [])))
    for i in range(n_ckpt):
        lab = labels[i] if i < len(labels) else f"ckpt{i}"
        db = _archetype_distribution(base, i)
        di = _archetype_distribution(impr, i)
        lines.append(f"| {lab} | {_fmt_dist(db, sum(db.values()))} | "
                     f"{_fmt_dist(di, sum(di.values()))} |")
    lines.append("")

    # Diagnostic findings
    fb = _findings_by_name(base)
    fi = _findings_by_name(impr)
    only_base = sorted(set(fb) - set(fi))
    only_impr = sorted(set(fi) - set(fb))
    shared = sorted(set(fi) & set(fb))
    lines.append("### Diagnostic findings")
    lines.append("")
    lines.append(f"- **Baseline only:** {', '.join(f'`{n}`' for n in only_base) or '_(none)_'}")
    lines.append(f"- **With-improvements only:** "
                 f"{', '.join(f'`{n}`' for n in only_impr) or '_(none)_'}")
    lines.append(f"- **Shared:** {', '.join(f'`{n}`' for n in shared) or '_(none)_'}")
    if only_impr:
        lines.append("")
        lines.append("Unique with-improvements findings (headline):")
        for n in only_impr:
            f = fi[n]
            sev = f.get("severity", "?")
            head = f.get("headline", "")
            lines.append(f"- `{n}` ({sev}): {head}")
    lines.append("")

    # Interaction range
    sb = _final_interaction_stats(base)
    si = _final_interaction_stats(impr)
    lines.append("### Final-checkpoint interaction range")
    lines.append("")
    if sb and si:
        lines.append("| Statistic | Baseline | With improvements |")
        lines.append("|---|---|---|")
        for k in ("min", "mean", "max"):
            lines.append(f"| {k} | {sb[k]:.4g} | {si[k]:.4g} |")
    else:
        lines.append("_(missing signatures in one or both reports)_")
    lines.append("")

    # Top-K feature overlap
    tb = _top_features(base, top_k)
    ti = _top_features(impr, top_k)
    jac = _jaccard(tb, ti)
    lines.append(f"### Top-{top_k} feature overlap")
    lines.append("")
    lines.append(f"- **Jaccard:** {jac:.2f} ({len(set(tb) & set(ti))} / "
                 f"{len(set(tb) | set(ti))})")
    lines.append(f"- **Baseline top-{top_k}:** "
                 f"{', '.join(f'`{x}`' for x in tb) or '_(none)_'}")
    lines.append(f"- **With-improvements top-{top_k}:** "
                 f"{', '.join(f'`{x}`' for x in ti) or '_(none)_'}")
    lines.append(f"- **Only in baseline:** "
                 f"{', '.join(f'`{x}`' for x in sorted(set(tb) - set(ti))) or '_(none)_'}")
    lines.append(f"- **Only in with-improvements:** "
                 f"{', '.join(f'`{x}`' for x in sorted(set(ti) - set(tb))) or '_(none)_'}")
    lines.append("")
    return "\n".join(lines)


def summarize_test(test_dir: pathlib.Path, top_k: int) -> Optional[pathlib.Path]:
    pairs = _find_pairs(test_dir)
    if not pairs:
        return None
    out_lines = [f"# Validation summary: `{test_dir.name}`", ""]
    out_lines.append(f"_{len(pairs)} baseline / with-improvements pair(s) found._")
    out_lines.append("")
    for stem, b, w in pairs:
        out_lines.append(_format_pair_section(stem, b, w, top_k))
    out_path = test_dir / "summary.md"
    out_path.write_text("\n".join(out_lines))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parent),
                    help="Root directory containing 0?_* test folders")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of top features for Jaccard overlap")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root {root!s} is not a directory")
    test_dirs = sorted(p for p in root.iterdir()
                       if p.is_dir() and p.name[:2].isdigit() and "_" in p.name)
    if not test_dirs:
        raise SystemExit(f"no 0?_* test directories under {root}")
    written = 0
    for td in test_dirs:
        out = summarize_test(td, args.top_k)
        if out is None:
            print(f"  [skip] {td.name}: no baseline/with pairs")
        else:
            print(f"  [ok]   {td.name}: {out.relative_to(root)}")
            written += 1
    print(f"Wrote {written} summary file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
