"""Validate an FFCA rulebook YAML against rulebook/schema.json.

Usage:
    python rulebook/validate.py rulebook/ffca_rules.yaml
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


def validate(rulebook_path: Path, schema_path: Path) -> int:
    schema = json.loads(schema_path.read_text())
    rulebook = yaml.safe_load(rulebook_path.read_text())

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(rulebook), key=lambda e: e.path)

    if errors:
        print(f"FAIL: {len(errors)} schema violation(s) in {rulebook_path.name}:")
        for err in errors:
            loc = ".".join(str(p) for p in err.path) or "(root)"
            print(f"  - at {loc}: {err.message}")
        return 1

    ids = [r["id"] for r in rulebook["rules"]]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        print(f"FAIL: duplicate rule ids: {dupes}")
        return 1

    by_kind = Counter(r["kind"] for r in rulebook["rules"])
    by_category = Counter(r["category"] for r in rulebook["rules"])
    by_severity = Counter(r.get("severity", "—") for r in rulebook["rules"])

    print(f"OK: {len(ids)} rules valid (v{rulebook['version']}).")
    print(f"  by kind:     {dict(by_kind)}")
    print(f"  by category: {dict(by_category)}")
    print(f"  by severity: {dict(by_severity)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    rb = Path(sys.argv[1])
    schema = Path(__file__).parent / "schema.json"
    sys.exit(validate(rb, schema))
