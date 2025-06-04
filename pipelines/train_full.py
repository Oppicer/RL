"""Full training pipeline: SFT -> GRM -> Alpha-Evolve.
This is heavily simplified and intended as a placeholder.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path):
    import yaml
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

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], load_in_4bit=True, device_map="auto")

    dataset = prepare_dataset(cfg)

    train_sft(cfg, model, tokenizer, dataset)

    # Placeholder for GRM-lite and Alpha-Evolve steps
    print("[INFO] GRM-lite and Alpha-Evolve steps would run here.")

if __name__ == "__main__":
    main()
