# Eval-Open-Source-LLMs

Evaluate open-source LLMs (Llama, Mistral, Qwen, DeepSeek, etc.) using [Inspect AI](https://inspect.aisi.org.uk/) for benchmarks, [W&B](https://wandb.ai/) for orchestration and tracking, and [CoreWeave](https://coreweave.com/) (via W&B Inference) for GPU-hosted model serving.

## Pipeline

```
1. Upload weights  ──>  2. Host on CoreWeave  ──>  3. Run evals
   (scripts/               (W&B Inference)          (Inspect AI +
    upload_weights.py)                                W&B Launch)
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

## Step 1: Upload Model Weights

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

## Step 2: Verify Model Endpoint

W&B Inference (backed by CoreWeave) serves models via an OpenAI-compatible API:

```bash
# List available models
python scripts/deploy_model.py --list

# Smoke-test a model
python scripts/deploy_model.py --model-id meta-llama/Llama-3.1-8B-Instruct
```

## Step 3: Run Evals

### Option A: Local Inspect AI (results auto-log to W&B)

```bash
# Run a predefined suite
python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --suite quick

# Run specific tasks
python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --tasks mmlu_0shot gsm8k

# Run all models against a suite
python scripts/run_evals.py --all-models --suite standard
```

### Option B: W&B Launch UI (CoreWeave-managed GPU)

W&B's built-in LLM Evaluation Jobs run benchmarks on CoreWeave GPUs with automatic leaderboard generation:

```bash
# Print step-by-step instructions
python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --mode launch
```

Or go directly to **W&B > Launch > Evaluate hosted API model** in the web UI.

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
  run_evals.py             # Step 3: Run evals (local or launch)
evals/
  example_inspect_task.py  # Example custom Inspect AI task
  example_custom_weave_scorers.py  # Example W&B Weave scorers
```
