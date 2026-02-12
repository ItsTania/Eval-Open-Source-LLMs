"""Step 1: Upload open-source model weights to W&B Model Registry.

Usage:
    # Upload a single model by HuggingFace repo ID
    python scripts/upload_weights.py --model-name meta-llama/Llama-3.1-8B-Instruct

    # Upload all models defined in configs/models.yaml
    python scripts/upload_weights.py --all

    # Upload LoRA weights from a local directory
    python scripts/upload_weights.py --model-name my-lora --lora-path ./lora-weights
"""

import argparse
import os
from pathlib import Path

import wandb
import yaml
from dotenv import load_dotenv

load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "eval-os-llms")
REGISTRY_NAME = "model"


def upload_hf_model(model_name: str, hf_repo: str, entity: str, project: str) -> str:
    """Upload a HuggingFace model reference to W&B Model Registry."""
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="upload-model",
        config={"model_name": model_name, "hf_repo": hf_repo},
    )

    artifact = wandb.Artifact(name=model_name, type="model")
    artifact.add_reference(f"hf://{hf_repo}")

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-{REGISTRY_NAME}/{model_name}",
    )

    print(f"Uploaded {hf_repo} as '{model_name}' to W&B Registry")
    run.finish()
    return logged_artifact.name


def upload_lora_weights(
    model_name: str, lora_path: str, base_model: str, entity: str, project: str
) -> str:
    """Upload LoRA weights to W&B with CoreWeave storage for inference."""
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="upload-lora",
        config={"model_name": model_name, "base_model": base_model},
    )

    artifact = wandb.Artifact(
        name=model_name,
        type="lora",
        metadata={"wandb.base_model": base_model},
        storage_region="coreweave-us",
    )
    artifact.add_dir(lora_path)

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    print(f"Uploaded LoRA weights '{model_name}' (base: {base_model}) to W&B")
    run.finish()
    return logged_artifact.name


def upload_all_from_config(config_path: str, entity: str, project: str):
    """Upload all models defined in the YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for model in config["models"]:
        name = model["name"]
        hf_repo = model["hf_repo"]
        model_type = model.get("type", "base")

        print(f"\n--- Uploading {name} ({hf_repo}) ---")

        if model_type == "lora":
            lora_path = model.get("lora_path")
            base_model = model.get("base_model")
            if not lora_path or not base_model:
                print(f"  Skipping {name}: lora_path and base_model required for LoRA")
                continue
            upload_lora_weights(name, lora_path, base_model, entity, project)
        else:
            upload_hf_model(name, hf_repo, entity, project)


def main():
    parser = argparse.ArgumentParser(description="Upload model weights to W&B Registry")
    parser.add_argument("--model-name", help="HuggingFace repo ID (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--artifact-name", help="Name for the W&B artifact (defaults to repo name)")
    parser.add_argument("--lora-path", help="Path to local LoRA weights directory")
    parser.add_argument("--base-model", help="Base model for LoRA (e.g. OpenPipe/Qwen3-14B-Instruct)")
    parser.add_argument("--all", action="store_true", help="Upload all models from configs/models.yaml")
    parser.add_argument("--config", default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity/team")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project name")
    args = parser.parse_args()

    if args.all:
        upload_all_from_config(args.config, args.entity, args.project)
    elif args.lora_path:
        if not args.base_model:
            parser.error("--base-model is required when uploading LoRA weights")
        name = args.artifact_name or Path(args.lora_path).name
        upload_lora_weights(name, args.lora_path, args.base_model, args.entity, args.project)
    elif args.model_name:
        name = args.artifact_name or args.model_name.split("/")[-1]
        upload_hf_model(name, args.model_name, args.entity, args.project)
    else:
        parser.error("Provide --model-name, --lora-path, or --all")


if __name__ == "__main__":
    main()
