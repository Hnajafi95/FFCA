"""Ask an LLM judge to score the semantic agreement of v0.6 determinism reruns.

The lexical Jaccard in `D_determinism/determinism.json` measures token
overlap, which conflates "same recommendation, different wording" with
"different recommendation". We ask Claude — with the rerun texts and no
FFCA prior context — to score semantic agreement on a 0-1 scale and to
list any genuine disagreements.

Output: D_determinism/llm_judge.json + a printed summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _judge_one_case(client, label: str, reruns: list[dict]) -> dict:
    """Send the reruns to Claude with a tight scoring instruction."""
    blocks = []
    for i, r in enumerate(reruns, 1):
        actions = "\n".join(f"  - {t}" for t in r.get("action_titles", []))
        obs = "\n".join(f"  - {o}" for o in r.get("rule_free_obs_titles", []))
        blocks.append(
            f"--- RERUN {i} ---\n"
            f"Executive summary:\n{r.get('exec_summary','').strip()}\n\n"
            f"Action titles:\n{actions or '  (none)'}\n\n"
            f"Rule-free observation titles:\n{obs or '  (none)'}"
        )
    user_prompt = (
        "Three reruns of an AI model-diagnosis tool, all produced from the "
        "EXACT same input. Your job: judge whether the three reruns are "
        "**semantically equivalent** in what they recommend, regardless of "
        "wording.\n\n"
        "Two diagnoses are semantically equivalent when:\n"
        "  - They name the same root issues (e.g., overfitting, data leakage),\n"
        "  - They recommend the same actions (e.g., 'stop training' and 'halt "
        "and regularize' count as the same action),\n"
        "  - They flag the same specific features by name.\n\n"
        "Two diagnoses are NOT equivalent when:\n"
        "  - They disagree on the root issue,\n"
        "  - One recommends an action the others don't (and that action is "
        "not implied by the others),\n"
        "  - They name different features as the key suspects.\n\n"
        f"Case: {label}\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "Respond in this exact JSON format inside <JUDGE>...</JUDGE> markers:\n"
        "<JUDGE>\n"
        "{\n"
        '  "semantic_agreement_score": 0.0-1.0,\n'
        '  "explanation": "one sentence on why",\n'
        '  "disagreements": ["short list of any genuine semantic disagreements"],\n'
        '  "stable_recommendations": ["list of recommendations that appeared in all 3 reruns, possibly worded differently"]\n'
        "}\n"
        "</JUDGE>"
    )
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1500,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content)
    # Parse fenced JSON
    import re
    m = re.search(r"<JUDGE>\s*(.*?)\s*</JUDGE>", text, re.DOTALL)
    if not m:
        return {"error": "no <JUDGE> fence in response", "raw": text[:400]}
    try:
        parsed = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        return {"error": f"bad json: {exc}", "raw": m.group(1)[:400]}
    parsed["usage"] = {
        "input_tokens": int(getattr(resp.usage, "input_tokens", 0)),
        "output_tokens": int(getattr(resp.usage, "output_tokens", 0)),
    }
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--determinism-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = Path(args.key_file).expanduser().read_text().strip()
    from anthropic import Anthropic
    client = Anthropic(api_key=key)

    det = json.loads(Path(args.determinism_json).read_text())

    out: dict[str, dict] = {}
    for label, v in det.items():
        if "error" in v or "reruns" not in v:
            out[label] = {"skipped": "no reruns present"}
            continue
        print(f"judging {label} ({len(v['reruns'])} reruns)...")
        out[label] = _judge_one_case(client, label, v["reruns"])
        score = out[label].get("semantic_agreement_score", "?")
        print(f"  → score: {score}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # Final headline
    scores = [v.get("semantic_agreement_score") for v in out.values()
              if isinstance(v.get("semantic_agreement_score"), (int, float))]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nMean semantic agreement across {len(scores)} cases: {avg:.2f}")
        print(f"(Compare to lexical Jaccard: ~0.4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
