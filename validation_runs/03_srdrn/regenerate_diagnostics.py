"""Re-generate the SRDRN α-FFCA report from the existing report.json so we
get the new Diagnostics narrative without re-running the full pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ffca import FFCAReport, FFCASignature
from ffca.improvements_pkg import CoSensitivityGroups, TrustScore


def reconstruct_signature(blob: dict) -> FFCASignature:
    sig = FFCASignature(
        impact=np.array(blob["impact"]),
        volatility=np.array(blob["volatility"]),
        nonlinearity=np.array(blob["nonlinearity"]),
        interaction=np.array(blob["interaction"]),
        feature_names=blob["feature_names"],
        archetypes=np.array(blob["archetypes"]) if blob.get("archetypes") is not None else None,
        interaction_ci=np.array(blob["interaction_ci"]) if blob.get("interaction_ci") is not None else None,
        metadata=blob.get("metadata", {}),
    )
    return sig


def regenerate(out_dir: Path):
    print(f"Regenerating {out_dir} …")
    blob = json.loads((out_dir / "report.json").read_text())
    sigs = [reconstruct_signature(b) for b in blob["signatures"]]
    labels = blob["checkpoint_labels"]
    # We can't reconstruct co-sensitivity guardrails without raw gradients.
    # But we can reconstruct trust from per-checkpoint signatures.
    trust = TrustScore() if len(sigs) >= 2 else None
    if trust:
        trust.compute(sigs, sigs[0].feature_names)

    # Co-Sensitivity diagnostics are in the existing JSON — preserve them
    # by constructing a stand-in object the report can read.
    cs = None
    cs_blob = blob.get("co_sensitivity")
    if cs_blob and cs_blob.get("groups"):
        cs = CoSensitivityGroups()
        cs.results = {int(k): v for k, v in cs_blob["groups"].items()}
        cs.diagnostics = cs_blob["diagnostics"]

    # Build a synthetic FFCAReport just for markdown generation
    class _Stub:
        pass
    rep = _Stub()
    rep.signatures = sigs
    rep.checkpoint_labels = labels
    rep.trust = trust
    rep.cosens = cs
    rep.timing = blob.get("timing", {})

    # Use the real FFCAReport.generate_markdown method
    from ffca.diagnostics import run_all
    rep.findings = run_all(sigs, labels, trust=trust, cosens=cs)
    rep.to_dict = lambda: blob  # not used here
    md = FFCAReport.generate_markdown(rep)
    (out_dir / "report.md").write_text(md)
    print(f"  ✓ {out_dir/'report.md'} ({len(md):,} chars)")
    print(f"  findings: {len(rep.findings)} "
          f"({sum(1 for f in rep.findings if f.severity=='critical')} critical, "
          f"{sum(1 for f in rep.findings if f.severity=='warn')} warn, "
          f"{sum(1 for f in rep.findings if f.severity=='info')} info)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    for d in ("alpha_with", "alpha_baseline"):
        if (base / d / "report.json").exists():
            regenerate(base / d)
