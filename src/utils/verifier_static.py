# src/utils/verifier_static.py
"""
Verifier for static diagnostic QA answers.

Given:
  - an EpisodicStore
  - a gold QA instance (optional, for consistency checks)
  - a model answer JSON (as dict)

It checks:
  - JSON structure (required keys)
  - provenance.fact_id exists in the store
  - provenance.features are valid feature names for that fact
  - provenance.file,row match fact metadata (if provided)
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional

from src.utils.episodic_store import EpisodicStore


REQUIRED_TOP_KEYS = {"direct_answer", "reasoning_answer", "provenance", "confidence"}
REQUIRED_PROV_KEYS = {"fact_id", "features", "file", "row"}


def verify_structure(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Check that all required keys are present and of roughly correct type."""
    issues = []

    missing = REQUIRED_TOP_KEYS - set(answer.keys())
    if missing:
        issues.append(f"Missing top-level keys: {sorted(missing)}")

    if "provenance" in answer:
        prov = answer["provenance"]
        if not isinstance(prov, dict):
            issues.append("provenance must be an object")
        else:
            missing_p = REQUIRED_PROV_KEYS - set(prov.keys())
            if missing_p:
                issues.append(f"Missing provenance keys: {sorted(missing_p)}")
            if "features" in prov and not isinstance(prov["features"], list):
                issues.append("provenance.features must be a list")

    conf = answer.get("confidence", None)
    if conf is None:
        issues.append("confidence is missing")
    else:
        try:
            c = float(conf)
            if not (0.0 <= c <= 1.0):
                issues.append("confidence must be between 0 and 1")
        except Exception:
            issues.append("confidence must be numeric")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
    }


def verify_provenance_against_store(
    answer: Dict[str, Any],
    store: EpisodicStore,
) -> Dict[str, Any]:
    """
    Check that:
      - provenance.fact_id exists
      - provenance.features are real features of that fact
      - provenance.file and row match fact metadata (if not None)
    """
    issues = []
    prov = answer.get("provenance", {})
    fact_id = prov.get("fact_id")
    features = prov.get("features", [])
    file_ = prov.get("file")
    row = prov.get("row")

    fact = None
    if fact_id is None:
        issues.append("provenance.fact_id is missing")
    else:
        fact = store.get_fact(str(fact_id))
        if fact is None:
            issues.append(f"provenance.fact_id '{fact_id}' not found in EpisodicStore")

    valid_feature_names = set()
    if fact is not None:
        valid_feature_names = {f["name"] for f in fact.get("features", [])}
        fact_file = fact.get("source_file")
        fact_row = fact.get("row_index")

        # check file/row if present in provenance
        if file_ is not None and fact_file is not None and file_ != fact_file:
            issues.append(f"provenance.file={file_} does not match fact.source_file={fact_file}")
        if row is not None and fact_row is not None and int(row) != int(fact_row):
            issues.append(f"provenance.row={row} does not match fact.row_index={fact_row}")

    # check features list
    invalid_feats = []
    for f in features or []:
        if f not in valid_feature_names:
            invalid_feats.append(f)
    if invalid_feats:
        issues.append(f"provenance.features contain unknown feature(s): {sorted(invalid_feats)}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
    }


def verify_answer(
    answer: Dict[str, Any],
    store: EpisodicStore,
) -> Dict[str, Any]:
    """
    Full verification entrypoint.

    Returns:
      {
        "structure_ok": bool,
        "provenance_ok": bool,
        "structure_issues": [...],
        "provenance_issues": [...]
      }
    """
    s = verify_structure(answer)
    p = verify_provenance_against_store(answer, store)

    return {
        "structure_ok": s["ok"],
        "provenance_ok": p["ok"],
        "structure_issues": s["issues"],
        "provenance_issues": p["issues"],
    }


# CLI: verify one answer JSON file
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify a model answer JSON against EpisodicStore.")
    parser.add_argument("--db", required=True, help="Path to episodic_store.db")
    parser.add_argument("--answer", required=True, help="Path to JSON file with model answer")
    args = parser.parse_args()

    with open(args.answer, "r") as f:
        ans = json.load(f)

    store = EpisodicStore(db_path=args.db)
    report = verify_answer(ans, store)
    store.close()

    print(json.dumps(report, indent=2))
