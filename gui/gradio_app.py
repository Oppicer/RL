"""Simple Gradio interface for the trained model."""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "outputs/sft_model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=generate, inputs="text", outputs="text")

if __name__ == "__main__":
    iface.launch()
