import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from human_eval.data import read_problems
from human_eval.execution import check_correctness
from human_eval.evaluation import write_jsonl


def generate_completion(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_humaneval(ckpt_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    samples = []
    for task_id, task in read_problems().items():
        completion = generate_completion(model, tokenizer, task["prompt"])
        samples.append({"task_id": task_id, "completion": completion})

    samples_file = out_dir / "humaneval_samples.jsonl"
    write_jsonl(samples_file, samples)
    results = check_correctness(str(samples_file), k=[1])
    with open(out_dir / "humaneval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("HumanEval results:", results)


def run_apps_dev_med(ckpt_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("minimario/apps_partial_sorted_0_300", split="train[:10]", trust_remote_code=True)
    correct = 0
    for row in ds:
        completion = generate_completion(model, tokenizer, row["problem"])
        if completion.strip() == row["full_sample"].strip():
            correct += 1
    acc = correct / len(ds)
    result = {"accuracy": acc}
    with open(out_dir / "apps_dev_med_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("APPS-Dev-Med accuracy:", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run code benchmarks")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--subset", type=str, default="humaneval,apps_dev_med")
    parser.add_argument("--out", type=str, default="eval_results")
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subset.split(",") if s.strip()]
    out_dir = Path(args.out)

    if "humaneval" in subsets:
        run_humaneval(args.ckpt, out_dir / "humaneval")
    if "apps_dev_med" in subsets:
        run_apps_dev_med(args.ckpt, out_dir / "apps_dev_med")
