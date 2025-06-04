"""Download a base model and prepare 4-bit weights for training.
This is a minimal example using HuggingFace Transformers and bitsandbytes.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

MODEL_NAME = "Qwen/Qwen1.5-1.8B"
OUTPUT_DIR = "models/4bit"

def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
