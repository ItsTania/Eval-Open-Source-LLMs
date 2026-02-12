"""Step 3: Run Inspect AI evals locally, hitting the W&B Inference endpoint.
                Results auto-log to W&B via inspect_wandb.

Usage:
    # Run evals locally via Inspect AI
    python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --suite quick

    # Run specific tasks
    python scripts/run_evals.py --model-id meta-llama/Llama-3.1-8B-Instruct --tasks mmlu_0shot gsm8k

    # Run all configured models against a suite
    python scripts/run_evals.py --all-models --suite standard

"""

import argparse
import os
import subprocess
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"


def load_config(config_path: str = "configs/eval_suites.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_models(config_path: str = "configs/models.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f)["models"]


def resolve_tasks(task_names: list[str], config: dict) -> list[dict]:
    """Resolve task names to their inspect_path and args from config."""
    tasks_config = config.get("tasks", {})
    resolved = []
    for name in task_names:
        if name not in tasks_config:
            print(f"Warning: Task '{name}' not found in config, skipping", file=sys.stderr)
            continue
        task = tasks_config[name]
        resolved.append({
            "name": name,
            "inspect_path": task["inspect_path"],
            "args": task.get("args", {}),
        })
    return resolved


def run_inspect_eval(model_id: str, task: dict, base_url: str) -> bool:
    """Run a single Inspect AI eval task against the W&B Inference endpoint."""
    model_spec = f"openai/{model_id}" # Uses the OpenAI API

    cmd = [
        "inspect", "eval", task["inspect_path"],
        "--model", model_spec,
        "--model-base-url", base_url,
        "-M", f"api_key={os.getenv('WANDB_API_KEY', '')}",
    ]

    for key, value in task["args"].items():
        cmd.extend(["-T", f"{key}={value}"])

    print(f"\n{'='*60}")
    print(f"Running: {task['name']}")
    print(f"Path:    {task['inspect_path']}")
    print(f"Args:    {task['args'] or '(none)'}")
    print(f"Model:   {model_id}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_local(model_ids: list[str], tasks: list[dict], base_url: str):
    """Run Inspect AI evals locally against hosted model endpoints."""
    all_results = {}
    for model_id in model_ids:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {model_id}")
        print(f"{'#'*60}")

        results = {}
        for task in tasks:
            success = run_inspect_eval(model_id, task, base_url)
            results[task["name"]] = "passed" if success else "failed"
            print(f"  {task['name']}: {'PASSED' if success else 'FAILED'}")

        all_results[model_id] = results

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_id, results in all_results.items():
        print(f"\n{model_id}:")
        for task_name, status in results.items():
            print(f"  {task_name}: {status}")

    print("\nResults have been logged to W&B via inspect_wandb.")
    print("Run 'inspect view' to browse detailed logs locally.")



def main():
    parser = argparse.ArgumentParser(description="Run evals against hosted models")
    parser.add_argument("--model-id", help="Model ID on W&B Inference")
    parser.add_argument("--all-models", action="store_true", help="Run against all models in config")
    parser.add_argument("--suite", help="Eval suite name (quick, standard, full)")
    parser.add_argument("--tasks", nargs="+", help="Specific task names to run")
    parser.add_argument("--base-url", default=WANDB_INFERENCE_BASE_URL, help="Model API base URL")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--suites-config", default="configs/eval_suites.yaml")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "eval-os-llms"))
    args = parser.parse_args()

    # Determine which models to evaluate
    if args.all_models:
        models = load_models(args.models_config)
        model_ids = [m["hf_repo"] for m in models]
    elif args.model_id:
        model_ids = [args.model_id]
    else:
        parser.error("Provide --model-id or --all-models")


    # Inspect tasks
    config = load_config(args.suites_config)

    if args.suite:
        suites = config.get("suites", {})
        if args.suite not in suites:
            available = ", ".join(suites.keys())
            print(f"Error: Unknown suite '{args.suite}'. Available: {available}", file=sys.stderr)
            sys.exit(1)
        task_names = suites[args.suite]["tasks"]
    elif args.tasks:
        task_names = args.tasks
    else:
        parser.error("Provide --suite or --tasks for local mode")

    tasks = resolve_tasks(task_names, config)
    if not tasks:
        print("Error: No valid tasks to run.", file=sys.stderr)
        sys.exit(1)

    run_local(model_ids, tasks, args.base_url)


if __name__ == "__main__":
    main()
