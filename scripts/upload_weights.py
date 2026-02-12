"""Step 1: Download open-source model weights and upload to W&B Model Registry.

Downloads model + tokenizer from HuggingFace, saves in a vLLM-compatible format,
then uploads the directory to W&B as an artifact.

Usage:
    # Download and upload a single model by HuggingFace repo ID
    python scripts/upload_weights.py --hf-repo Qwen/Qwen2.5-7B-Instruct

    # Specify a custom artifact name and save directory
    python scripts/upload_weights.py --hf-repo Qwen/Qwen2.5-7B-Instruct \
        --artifact-name qwen-2.5-7b --save-dir ./weights/qwen-2.5-7b

    # Upload all models defined in configs/models.yaml
    python scripts/upload_weights.py --all

    # Upload LoRA weights from a local directory
    python scripts/upload_weights.py --artifact-name my-lora --lora-path ./lora-weights \
        --base-model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
from pathlib import Path

import wandb
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "eval-os-llms")
REGISTRY_NAME = "model"
DEFAULT_SAVE_ROOT = Path("weights")


def download_and_save(hf_repo: str, save_dir: Path) -> Path:
    """Download model + tokenizer from HuggingFace and save locally."""
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer from {hf_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    tokenizer.save_pretrained(save_dir)

    print(f"Downloading model from {hf_repo}...")
    model = AutoModelForCausalLM.from_pretrained(hf_repo)
    model.save_pretrained(save_dir)

    print(f"Saved to {save_dir}")
    return save_dir


def upload_model(
    artifact_name: str, save_dir: Path, hf_repo: str, entity: str, project: str
) -> str:
    """Upload saved model weights to W&B Model Registry."""
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="upload-model",
        config={"artifact_name": artifact_name, "hf_repo": hf_repo},
    )

    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_dir(str(save_dir))

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-{REGISTRY_NAME}/{artifact_name}",
    )

    print(f"Uploaded '{artifact_name}' to W&B Registry")
    run.finish()
    return logged_artifact.name


def upload_lora_weights(
    artifact_name: str, lora_path: str, base_model: str, entity: str, project: str
) -> str:
    """Upload LoRA weights to W&B with CoreWeave storage for inference."""
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="upload-lora",
        config={"artifact_name": artifact_name, "base_model": base_model},
    )

    artifact = wandb.Artifact(
        name=artifact_name,
        type="lora",
        metadata={"wandb.base_model": base_model},
        storage_region="coreweave-us",
    )
    artifact.add_dir(lora_path)

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    print(f"Uploaded LoRA weights '{artifact_name}' (base: {base_model}) to W&B")
    run.finish()
    return logged_artifact.name


def upload_all_from_config(config_path: str, entity: str, project: str):
    """Download and upload all models defined in the YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for model in config["models"]:
        name = model["name"]
        hf_repo = model["hf_repo"]
        model_type = model.get("type", "base")

        print(f"\n--- {name} ({hf_repo}) ---")

        if model_type == "lora":
            lora_path = model.get("lora_path")
            base_model = model.get("base_model")
            if not lora_path or not base_model:
                print(f"  Skipping {name}: lora_path and base_model required for LoRA")
                continue
            upload_lora_weights(name, lora_path, base_model, entity, project)
        else:
            save_dir = DEFAULT_SAVE_ROOT / name
            download_and_save(hf_repo, save_dir)
            upload_model(name, save_dir, hf_repo, entity, project)


def main():
    parser = argparse.ArgumentParser(
        description="Download HF model weights and upload to W&B Registry"
    )
    parser.add_argument("--hf-repo", help="HuggingFace repo ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--artifact-name", help="Name for the W&B artifact (defaults to repo name)")
    parser.add_argument("--save-dir", help="Local directory to save weights (defaults to weights/<name>)")
    parser.add_argument("--lora-path", help="Path to local LoRA weights directory")
    parser.add_argument("--base-model", help="Base model for LoRA (e.g. Qwen/Qwen2.5-7B-Instruct)")
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
    elif args.hf_repo:
        name = args.artifact_name or args.hf_repo.split("/")[-1]
        save_dir = Path(args.save_dir) if args.save_dir else DEFAULT_SAVE_ROOT / name
        download_and_save(args.hf_repo, save_dir)
        upload_model(name, save_dir, args.hf_repo, args.entity, args.project)
    else:
        parser.error("Provide --hf-repo, --lora-path, or --all")


if __name__ == "__main__":
    main()
