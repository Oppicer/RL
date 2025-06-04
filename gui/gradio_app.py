"""Simple Gradio interface for the trained model."""

import json
from pathlib import Path

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "outputs/sft_model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

BEST_INFO_PATH = Path("outputs/dgm_best.json")


def best_summary() -> str:
    if BEST_INFO_PATH.exists():
        data = json.loads(BEST_INFO_PATH.read_text())
        return f"Best DGM agent: {data['agent_id']} (score {data['score']:.3f})"
    return "Best DGM agent: N/A"


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=generate,
    inputs="text",
    outputs="text",
    description=best_summary(),
)

if __name__ == "__main__":
    iface.launch()
