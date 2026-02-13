# IndustryAssetEQA

**Embodied Question Answering for Industrial Asset Maintenance**

This repository implements **IndustryAssetEQA** — a neurosymbolic embodied QA system that grounds answers in episode-level telemetry, an ISO-derived Failure Mode and Effects Analysis Knowledge Graph (FMEA-KG), and a causal risk simulator to support evidence-grounded, counterfactual, and action-oriented maintenance QA.

Compared to LLM-only baselines, IndustryAssetEQA substantially improves structural validity, provenance accuracy, counterfactual reasoning reliability, and reduces unsafe expert-rated overclaims.

The repository includes:

- Inference scripts for large language models  
- Episodic memory (SQLite-backed store)  
- FMEA knowledge graph resources  
- Counterfactual risk simulator  
- Evaluation and ablation pipelines  
- Structured episode datasets  

---

## Quick Start (TL;DR)

1. Install dependencies (recommend a venv).
2. Set your model API credentials (`OPENAI_API_KEY`, `BASE_URL` if using non-default endpoints).
3. Run inference (example):

```bash
python -m src.scripts.run_inference_full --start 0 --end 100
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## API Configuration

IndustryAssetEQA uses black-box API access to LLMs (e.g., GPT-4o-mini, Claude Sonnet 4).

Set required environment variables:

```bash
export OPENAI_API_KEY="your_key_here"
export API_KEY="your_key_here"
export BASE_URL="https://..."   # optional if using custom endpoint
```

Default decoding settings:

- `temperature = 0.0`
- JSON output enforced

---

## Running Inference

Primary script:

```
src/scripts/run_inference_full.py
```

Basic usage:

```bash
python -m src.scripts.run_inference_full --start 0 --end 100
```

Available arguments:

- `--start` → start index (inclusive)
- `--end` → end index (exclusive)
- `--max` → maximum number of items to process

Example full run:

```bash
python -m src.scripts.run_inference_full --start 0 --end 5716
```

---

## Output Format

Predictions are written as JSONL.

Successful prediction:

```json
{"qa_id": "...", "answer": {...}}
```

Error record:

```json
{"qa_id": "...", "error": "..."}
```

Resume support is enabled: previously processed `qa_id`s are automatically skipped.

---

## Prompt and Output Contract

Each model receives:

- Task specification  
- Structured episode-level fact  
- Optional FMEA-KG context  
- Strict JSON output schema  

Expected output format:

```json
{
  "direct_answer": "...",
  "reasoning_answer": "...",
  "provenance": {...},
  "confidence": ...
}
```

Counterfactual tasks include:

```json
"counterfactual": {
  "risk_before": ...,
  "risk_after": ...,
  "delta_risk": ...,
  "direction": ...
}
```

---

## Evaluation

Metrics include:

- Structural Validity (`Struct.OK`)
- Provenance Accuracy (`Prov.OK`)
- Label Consistency
- Counterfactual Direction Accuracy
- Entailment Pass Rate
- Claim Precision
- Full Pass Rate

Example evaluation command:

```bash
python -m src.eval.evaluate_predictions \
  --preds data/outputs/pdm/preds_pdm_qas_diagnostic.jsonl \
  --gold data/outputs/pdm/pdm_qas_diagnostic.jsonl
```

---

## Ablation Experiments

Supported ablations:

- No episodic memory  
- No FMEA-KG  
- No provenance enforcement  
- No risk simulator  

Example:

```bash
python -m src.scripts.run_inference_full --disable-kg
```

(Check script flags for exact CLI options.)

---

## Datasets

The system supports:

- Microsoft Azure Predictive Maintenance (PdM)
- NASA C-MAPSS turbofan engines
- Genesis cyber-physical production system
- Hydraulic systems condition monitoring

Episodes are encoded as structured, time-situated facts with full provenance.

---

## Reproducing Paper Results

1. Ensure episodic databases exist:
   - `pdm_episodic_store.db`
   - `episodic_store_hyd.db`
2. Run inference for each QA dataset.
3. Run evaluation scripts.
4. Aggregate metrics.

Example canonical runs:

```bash
# PDM diagnostic
python -m src.scripts.run_inference_full --start 0 --end 5716

# PDM counterfactual
python -m src.scripts.run_inference_full --start 0 --end 761

# Hydraulic diagnostic
python -m src.scripts.run_inference_full --start 0 --end 2205

# Hydraulic counterfactual
python -m src.scripts.run_inference_full --start 0 --end 2184
```

---

## Troubleshooting

**Fact not found in EpisodicStore**
- Verify `DB_PATH` is correct.
- Ensure QA `fact_id` matches stored episodes.

**API rate limits**
- Increase backoff delay.
- Reduce batch size.

**JSON parsing errors**
- Inspect failed entries in output JSONL.
- Ensure model respects JSON contract.


---

## Citation

If you use this repository, please cite:

```
@inproceedings{industryasseteqa2026,
  title={IndustryAssetEQA: Neurosymbolic Embodied Question Answering for Industrial Asset Maintenance},
  author={...},
  year={2026}
}
```

---

## Contact

For issues or questions, please open a GitHub issue.
