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
    parser.add_argument(
        "--no_dgm",
        action="store_true",
        help="Skip the lightweight DGM loop after Alpha-Evolve",
    )
    parser.add_argument(
        "--export_fp16",
        action="store_true",
        help="Merge LoRA weights and export an FP16 model",
    )
    parser.add_argument(
        "--export_awq",
        action="store_true",
        help="Quantize the merged model to AWQ int4 format",
    )
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


def export_fp16(model, tokenizer, out_dir="outputs/fp16_model"):
    """Merge LoRA weights and save an FP16 model."""
    try:
        merged = model.merge_and_unload()
    except AttributeError:
        # If the model is already merged, proceed.
        merged = model
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def export_awq(src_dir, out_dir="outputs/awq_model"):
    """Quantize a model directory to AWQ int4 format."""
    try:
        from awq import AutoAWQForCausalLM
    except Exception:
        print("[WARN] AutoAWQ not installed; skipping AWQ export.")
        return None

    model = AutoAWQForCausalLM.from_pretrained(src_dir)
    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    model.quantize(tokenizer, out_dir)
    print(f"[INFO] AWQ model saved to {out_dir}")
    return out_dir


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], load_in_4bit=True, device_map="auto")

    dataset = prepare_dataset(cfg)

    train_sft(cfg, model, tokenizer, dataset)

    if args.export_fp16 or args.export_awq:
        fp16_dir = export_fp16(model, tokenizer)
        if args.export_awq:
            export_awq(fp16_dir)

    # Placeholder for GRM-lite and Alpha-Evolve steps
    print("[INFO] GRM-lite and Alpha-Evolve steps would run here.")

    if not args.no_dgm:
        from dgm.loop import run as dgm_run
        dgm_run()

if __name__ == "__main__":
    main()
