# scripts/run_inference_static.py

import json
from src.utils.episodic_store import EpisodicStore
from src.utils.prompt_builder_static import load_qa_index, build_prompt_for_qa
import os
import json
from openai import OpenAI

# Create a single global client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DB_PATH = "data/episodic_store.db"
QA_PATH = "data/c_qa.jsonl"
OUT_PATH = "data/c_preds.jsonl"

def call_llm_and_parse_json(system: str, user: str) -> dict:
    """
    Call the LLM and parse a JSON object with keys:
      direct_answer, reasoning_answer, provenance, confidence
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or another model you prefer
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # Basic sanity check: make sure required keys exist
    required_keys = {"direct_answer", "reasoning_answer", "provenance", "confidence"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"LLM output missing keys {missing}. Got: {data}")

    return data


def main():
    store = EpisodicStore(DB_PATH)
    qa_index = load_qa_index(QA_PATH)

    # Get just ONE (qa_id, qa) pair
    try:
        qa_id, qa = next(iter(qa_index.items()))
    except StopIteration:
        print("No QA items found in the index.")
        store.close()
        return

    prompts = build_prompt_for_qa(store, qa)
    model_json = call_llm_and_parse_json(
        system=prompts["system"],
        user=prompts["user"],
    )

    # Write ONLY this one result
    with open(OUT_PATH, "w") as fout:
        fout.write(json.dumps({"qa_id": qa_id, "answer": model_json}) + "\n")

    store.close()
    print(f"Wrote predictions for a single QA item to {OUT_PATH}")


if __name__ == "__main__":
    main()
