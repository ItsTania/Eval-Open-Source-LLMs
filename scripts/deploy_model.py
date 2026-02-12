"""Step 2: Verify model is hosted via W&B Inference (CoreWeave).

W&B Inference exposes an OpenAI-compatible API at https://api.inference.wandb.ai/v1
backed by CoreWeave GPUs. This script lists available models and runs a smoke test.

Usage:
    # List all available models on W&B Inference
    python scripts/deploy_model.py --list

    # Smoke-test a specific model
    python scripts/deploy_model.py --model-id meta-llama/Llama-3.1-8B-Instruct

    # Test a LoRA artifact
    python scripts/deploy_model.py --model-id "wandb-artifact:///my-team/my-project/my-lora:latest"
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def get_client(entity: str = None, project: str = None) -> OpenAI:
    """Create an OpenAI client pointed at W&B Inference."""
    kwargs = {
        "api_key": WANDB_API_KEY,
        "base_url": WANDB_INFERENCE_BASE_URL,
    }
    if entity and project:
        kwargs["project"] = f"{entity}/{project}"
    return OpenAI(**kwargs)


def list_models(client: OpenAI):
    """List all models available on W&B Inference."""
    print("Available models on W&B Inference:\n")
    models = client.models.list()
    for model in models.data:
        print(f"  {model.id}")
    print(f"\nTotal: {len(models.data)} models")


def smoke_test(client: OpenAI, model_id: str):
    """Run a quick completion to verify the model endpoint works."""
    print(f"Smoke-testing model: {model_id}\n")

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": "What is 2 + 2? Answer in one word."},
        ],
        max_tokens=32,
        temperature=0.0,
    )

    reply = response.choices[0].message.content
    usage = response.usage

    print(f"Response: {reply}")
    print(f"Tokens — prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}")
    print(f"\nModel endpoint is working. Use this for evals:")
    print(f"  --model-id {model_id}")
    print(f"  --base-url {WANDB_INFERENCE_BASE_URL}")


def main():
    parser = argparse.ArgumentParser(description="Verify model hosting via W&B Inference")
    parser.add_argument("--model-id", help="Model ID to smoke-test")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"), help="W&B entity/team")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "eval-os-llms"), help="W&B project")
    args = parser.parse_args()

    if not WANDB_API_KEY:
        print("Error: WANDB_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    client = get_client(args.entity, args.project)

    if args.list:
        list_models(client)
    elif args.model_id:
        smoke_test(client, args.model_id)
    else:
        list_models(client)


if __name__ == "__main__":
    main()
