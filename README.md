# Eval-Open-Source-LLMs

Evaluate open-source LLMs (Llama, Mistral, Qwen, DeepSeek, etc.) using [Inspect AI](https://inspect.aisi.org.uk/) for benchmarks, [W&B](https://wandb.ai/) for orchestration and tracking, and [CoreWeave](https://coreweave.com/) (via W&B Inference) for GPU-hosted model serving.

## Pipeline

**Option 1:** Upload weights and run evals via W&B UI
```
1. Upload weights  ──>  Run eval job via W&B UI
   (scripts/
    upload_weights.py)
```

**Option 2:** Use W&B Inference endpoint and run evals locally
```
2. Verify endpoint  ──>  3. Run evals locally
   (scripts/                (scripts/
    deploy_model.py)         run_evals.py)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
wandb login
```

Required environment variables (`.env`):
- `WANDB_API_KEY` — your W&B API key
- `WANDB_ENTITY` — your W&B team/entity
- `WANDB_PROJECT` — project name (default: `eval-os-llms`)
- `HF_TOKEN` — HuggingFace token (for gated models like Llama)

## Running Evals

There are two ways to run evals:

1. Use W&B Inference (backed by CoreWeave) to serves models via an OpenAI-compatible API then run the model locally. (Script 1)

2. Upload model weight (after training etc etc) to wandb as a VLLM artifact then run a evaluation job on W&B (backed by CoreWeave)

## Script 1: Upload Model Weights

Upload open-source model weights from HuggingFace to the W&B Model Registry:

```bash
# Single model
python scripts/upload_weights.py --model-name meta-llama/Llama-3.1-8B-Instruct

# All models from config
python scripts/upload_weights.py --all

# LoRA weights (stored on CoreWeave for inference)
python scripts/upload_weights.py --lora-path ./my-lora --base-model OpenPipe/Qwen3-14B-Instruct
```

Models are defined in `configs/models.yaml`.

## Script 2: Verify Model Endpoint

W&B Inference (backed by CoreWeave) serves models via an OpenAI-compatible API:

```bash
# List available models
python scripts/deploy_model.py --list

# Smoke-test a model
python scripts/deploy_model.py --model-id meta-llama/Llama-3.1-8B-Instruct
```

## Script 3: Run Evals (results auto-log to W&B)

```bash
# Run a predefined suite
python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --suite quick

# Run specific tasks
python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --tasks mmlu_0shot gsm8k

# Run all models against a suite
python scripts/run_evals.py --all-models --suite standard
```

## Configuration

### `configs/models.yaml` — Models to evaluate
```yaml
models:
  - name: llama-3.1-8b-instruct
    hf_repo: meta-llama/Llama-3.1-8B-Instruct
    type: base
```

### `configs/eval_suites.yaml` — Tasks and suites

Tasks map to Inspect Evals packages with optional arguments:
```yaml
tasks:
  mmlu_0shot:
    inspect_path: inspect_evals/mmlu
    args: {}
  gsm8k:
    inspect_path: inspect_evals/gsm8k
    args: {}

suites:
  quick:
    tasks: [mmlu_0shot, gsm8k]
```

## Project Structure

```
configs/
  models.yaml              # Model definitions
  eval_suites.yaml         # Task definitions + eval suites
scripts/
  upload_weights.py        # Step 1: Upload to W&B Registry
  deploy_model.py          # Step 2: Verify W&B Inference endpoint
  run_evals.py             # Step 3: Run evals
evals/
  example_inspect_task.py  # Example custom Inspect AI task
  example_custom_weave_scorers.py  # Example W&B Weave scorers
```
