# IndustryAssetEQA

**Embodied Question Answering for Industrial Asset Maintenance**

This repository implements **IndustryAssetEQA** — a neurosymbolic embodied QA system that grounds answers in episode-level telemetry, an ISO-derived FMEA knowledge graph, and a causal risk simulator to support evidence-grounded, counterfactual, and action-oriented maintenance QA. The repo includes code to run inference with large language models, evaluation scripts, datasets (structured episode facts and QA tasks), and KG resources.

---

## Quick start (TL;DR)

1. Install dependencies (recommend a venv).
2. Set your model API credentials (`OPENAI_API_KEY`, `BASE_URL` if using non-default endpoints).
3. Run inference (example):  
   ```bash
   python -m src.scripts.run_inference_full --start 0 --end 100
