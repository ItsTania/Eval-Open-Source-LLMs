"""Step 1: Upload open-source model weights to W&B Model Registry.

Usage:
    # Upload from an S3 bucket
    python scripts/upload_weights.py --model-name Llama-3.1-8B --reference s3://my-bucket/models/llama-3.1-8b

    # Upload from a GCS bucket
    python scripts/upload_weights.py --model-name Llama-3.1-8B --reference gs://my-bucket/models/llama-3.1-8b

    # Upload from an HTTP(S) URL
    python scripts/upload_weights.py --model-name Llama-3.1-8B --reference https://example.com/models/llama-3.1-8b

    # Upload from a local directory
    python scripts/upload_weights.py --model-name Llama-3.1-8B --reference /data/models/llama-3.1-8b

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


_SUPPORTED_SCHEMES = ("s3://", "gs://", "http://", "https://", "file://")


def _validate_reference(reference: str) -> str:
    """Validate that reference uses a supported scheme or is a local path."""
    if any(reference.startswith(scheme) for scheme in _SUPPORTED_SCHEMES):
        return reference
    local_path = Path(reference)
    if local_path.exists():
        return str(local_path.resolve())
    raise ValueError(
        f"Invalid reference: {reference!r}. "
        f"Must be a supported URI ({', '.join(_SUPPORTED_SCHEMES)}) or an existing local path."
    )


def upload_model(model_name: str, reference: str, entity: str, project: str) -> str:
    """Upload a model reference (S3 or local path) to W&B Model Registry."""
    reference = _validate_reference(reference)

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="upload-model",
        config={"model_name": model_name, "reference": reference},
    )

    artifact = wandb.Artifact(name=model_name, type="model")
    artifact.add_reference(reference, name=model_name)

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-{REGISTRY_NAME}/{model_name}",
    )

    print(f"Uploaded {reference} as '{model_name}' to W&B Registry")
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
        reference = model["reference"]
        model_type = model.get("type", "base")

        print(f"\n--- Uploading {name} ({reference}) ---")

        if model_type == "lora":
            lora_path = model.get("lora_path")
            base_model = model.get("base_model")
            if not lora_path or not base_model:
                print(f"  Skipping {name}: lora_path and base_model required for LoRA")
                continue
            upload_lora_weights(name, lora_path, base_model, entity, project)
        else:
            upload_model(name, reference, entity, project)


def main():
    parser = argparse.ArgumentParser(description="Upload model weights to W&B Registry")
    parser.add_argument("--model-name", help="Name for the W&B artifact")
    parser.add_argument("--reference", help="Model reference: S3 (s3://), GCS (gs://), HTTP(S), file://, or local path")
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
        name = args.model_name or Path(args.lora_path).name
        upload_lora_weights(name, args.lora_path, args.base_model, args.entity, args.project)
    elif args.reference:
        if not args.model_name:
            parser.error("--model-name is required when uploading a model reference")
        upload_model(args.model_name, args.reference, args.entity, args.project)
    else:
        parser.error("Provide --reference with --model-name, --lora-path, or --all")


if __name__ == "__main__":
    main()
