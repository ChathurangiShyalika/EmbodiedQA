# src/utils/prompt_builder_static.py
"""
Prompt builder for static and time-series QA over EpisodicStore.

Given:
  - episodic_store.db
  - a QA instance (e.g. from c_qa.jsonl or pdm_qa_cf.jsonl)
it builds a prompt that:
  - presents the evidence (features) for the fact
  - asks the question
  - instructs the LLM to answer in strict JSON.

Two main task types:
  - diagnostic (default): keys = direct_answer, reasoning_answer, provenance, confidence
  - counterfactual: keys = direct_answer, reasoning_answer, provenance, counterfactual, confidence

CLI example:

python -m src.utils.prompt_builder_static \
  --db data/episodic_store.db \
  --qa data/c_qa.jsonl \
  --qa-id usm_c_c_0
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional

from src.utils.episodic_store import EpisodicStore


def load_qa_index(qa_path: str) -> Dict[str, Dict[str, Any]]:
    """Load QA JSONL and index by qa_id."""
    idx: Dict[str, Dict[str, Any]] = {}
    with open(qa_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qa_id")
            if not qid:
                continue
            idx[qid] = obj
    return idx


def format_evidence_block(fact: Dict[str, Any], max_features: int = 20) -> str:
    asset_id = fact.get("asset_id", "asset")
    label = fact.get("label", "Unknown")
    fact_id = fact.get("fact_id", "unknown_fact")
    source_file = fact.get("source_file", "unknown")
    row_index = fact.get("row_index", "unknown")

    # Optional time-window fields for TS facts (PdM)
    start_time = fact.get("start_time")
    end_time = fact.get("end_time")

    features = fact.get("features", [])[:max_features]

    lines: List[str] = []
    lines.append(f"fact_id: {fact_id}")
    lines.append(f"asset_id: {asset_id}")
    lines.append(f"source_file: {source_file}")
    lines.append(f"row_index: {row_index}")

    if start_time and end_time:
        lines.append(f"window_start: {start_time}")
        lines.append(f"window_end: {end_time}")

    lines.append(f"dataset_label: {label}")
    lines.append("")
    lines.append("diagnostic_features:")
    for feat in features:
        name = feat.get("name")
        val = feat.get("value")
        lines.append(f"  - {name}: {val}")

    return "\n".join(lines)


def build_prompt_for_qa(
    store: EpisodicStore,
    qa: Dict[str, Any],
    system_role: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a prompt (system + user) for a single QA instance.

    Returns:
      {"system": system_prompt, "user": user_prompt}
    """
    fact_id = qa["fact_id"]
    fact = store.get_fact(fact_id)
    if fact is None:
        raise ValueError(f"Fact {fact_id} not found in EpisodicStore.")

    evidence_block = format_evidence_block(fact)
    question = qa["question"]
    task_type = qa.get("task_type", "diagnostic")

    # ------------------------
    # System prompt
    # ------------------------
    if task_type == "counterfactual":
        system_prompt = system_role or (
            "You are an industrial maintenance and reliability analyst. "
            "You MUST base your reasoning only on the provided evidence and the hypothetical "
            "intervention implied by the question. Do NOT invent features or facts that are not "
            "in the evidence. Always return a single JSON object with keys: "
            "direct_answer, reasoning_answer, provenance, counterfactual, confidence."
        )
    else:
        # default: diagnostic / static QA
        system_prompt = system_role or (
            "You are an industrial diagnostic assistant. "
            "You MUST base your reasoning only on the provided evidence. "
            "Do NOT invent features or facts that are not in the evidence. "
            "Always return a single JSON object with keys: "
            "direct_answer, reasoning_answer, provenance, confidence."
        )

    # ------------------------
    # User prompt
    # ------------------------
    if task_type == "counterfactual":
        # Counterfactual QA: ask about how risk changes under an intervention.
        user_prompt = f"""You are given evidence for a single machine episode (time window or snapshot).

EVIDENCE:
{evidence_block}

QUESTION:
{question}

Your task:
- Decide whether the risk of failure for this episode would increase, decrease, or stay roughly the same
  under the hypothetical change described in the question.
- Ground your reasoning in the provided features (e.g., maintenance age, telemetry aggregates, etc.).

RESPONSE FORMAT (VERY IMPORTANT):
Return ONLY a single valid JSON object with the following keys:

- "direct_answer": a short sentence answering the question, e.g. "The failure risk would decrease."
- "reasoning_answer": a more detailed explanation citing specific feature names and values from the evidence.
- "provenance": a JSON object with:
    - "fact_id": the fact_id from the evidence,
    - "features": a list of feature names you actually used in reasoning,
    - "file": the source_file from the evidence,
    - "row": the row_index from the evidence.
- "counterfactual": a JSON object summarizing your view of the intervention's effect. It MUST include:
    - "direction": one of "increase", "decrease", or "no_change".
  Optionally, you may also include:
    - "risk_before": your estimate of baseline risk (0–1, optional),
    - "risk_after": your estimate after the intervention (0–1, optional),
    - "delta_risk": your estimate of risk_after - risk_before (optional).
- "confidence": a number between 0 and 1 indicating your confidence in this counterfactual judgment.

Example JSON skeleton (do NOT copy the content, only the structure):
{{
  "direct_answer": "The failure risk would decrease.",
  "reasoning_answer": "Because the hours_since_last_maint_comp3 would drop from 477 to 0, which is associated with much lower failure rates in similar episodes.",
  "provenance": {{
    "fact_id": "pdm_m56_comp3_2015-01-02T03",
    "features": ["hours_since_last_maint_comp3", "vibration_max"],
    "file": "PdM_telemetry.csv",
    "row": 0
  }},
  "counterfactual": {{
    "direction": "decrease",
    "risk_before": 1.0,
    "risk_after": 0.01,
    "delta_risk": -0.99
  }},
  "confidence": 0.9
}}

Now produce your JSON answer.
"""
    else:
        # Diagnostic QA: original behavior
        user_prompt = f"""You are given diagnostic evidence for a single episode (e.g., sensor snapshot or time window).

EVIDENCE:
{evidence_block}

QUESTION:
{question}

RESPONSE FORMAT (VERY IMPORTANT):
Return ONLY a single valid JSON object with the following keys:

- "direct_answer": a short answer to the question in 1–2 sentences.
- "reasoning_answer": a more detailed explanation citing specific feature names and values from the evidence.
- "provenance": a JSON object with:
    - "fact_id": the fact_id from the evidence,
    - "features": a list of feature names you actually used in reasoning,
    - "file": the source_file from the evidence,
    - "row": the row_index from the evidence.
- "confidence": a number between 0 and 1 indicating your confidence.

Example JSON skeleton (do NOT copy the content, only the structure):
{{
  "direct_answer": "...",
  "reasoning_answer": "...",
  "provenance": {{
    "fact_id": "c_0",
    "features": ["feature1", "feature2"],
    "file": "c.csv",
    "row": 0
  }},
  "confidence": 0.8
}}

Now produce your JSON answer.
"""

    return {"system": system_prompt, "user": user_prompt}


# CLI helper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LLM prompt for a QA instance.")
    parser.add_argument("--db", required=True, help="Path to episodic_store.db")
    parser.add_argument("--qa", required=True, help="Path to QA JSONL file (e.g. c_qa.jsonl or pdm_qa_cf.jsonl)")
    parser.add_argument("--qa-id", required=True, help="qa_id to build prompt for")
    args = parser.parse_args()

    store = EpisodicStore(db_path=args.db)
    qa_index = load_qa_index(args.qa)
    qa = qa_index.get(args.qa_id)
    if qa is None:
        raise SystemExit(f"qa_id {args.qa_id} not found in {args.qa}")

    prompts = build_prompt_for_qa(store, qa)
    print("=== SYSTEM PROMPT ===")
    print(prompts["system"])
    print("\n=== USER PROMPT ===")
    print(prompts["user"])
    store.close()
