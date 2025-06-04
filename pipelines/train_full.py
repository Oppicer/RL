"""Full training pipeline: SFT -> GRM -> Alpha-Evolve.
This is heavily simplified and intended as a placeholder.
"""

import argparse
from pathlib import Path

import random
import torch
import numpy as np
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_run(cfg, seed: int):
    """Log configuration and seed to outputs directory."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    log_file = out_dir / "run.log"
    # Append run information as a single YAML document
    log_data = {"config": cfg, "seed": seed}
    with open(log_file, "a") as f:
        yaml.safe_dump(log_data, f, explicit_start=True)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_dataset(cfg):
    ds = load_dataset("text", data_files=cfg["dataset"], split="train")
    return ds


def train_sft(cfg, model, tokenizer, dataset):
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"])
    model = get_peft_model(model, lora)

    training_args = TrainingArguments(
        output_dir="outputs/sft",
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        gradient_checkpointing=True,
        fp16=True,
        logging_steps=10,
        save_steps=500,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained("outputs/sft_model")
    tokenizer.save_pretrained("outputs/sft_model")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    log_run(cfg, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], load_in_4bit=True, device_map="auto")

    dataset = prepare_dataset(cfg)

    train_sft(cfg, model, tokenizer, dataset)

    # Placeholder for GRM-lite and Alpha-Evolve steps
    print("[INFO] GRM-lite and Alpha-Evolve steps would run here.")

if __name__ == "__main__":
    main()
