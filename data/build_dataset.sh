#!/usr/bin/env bash
# Build the training dataset by fetching open-source code datasets and performing basic cleaning.
# NOTE: This is a lightweight placeholder that shows how data could be prepared.

set -euo pipefail

DATA_DIR="${DATASET_DIR:-$(pwd)/dataset}"
mkdir -p "$DATA_DIR"
export DATA_DIR

# Example: fetch a small subset of The Stack v2 from HuggingFace
python - <<'PY'
import datasets
import os

dataset = datasets.load_dataset('bigcode/the-stack-dedup', data_dir='data/python', split='train', streaming=True)
# Keep only a small sample for demonstration
sample = dataset.take(1000)

os.makedirs(os.environ.get('DATA_DIR', 'dataset'), exist_ok=True)
with open(os.path.join(os.environ.get('DATA_DIR', 'dataset'), 'sample.txt'), 'w') as f:
    for row in sample:
        f.write(row['content'] + '\n')
PY

# Tokenise with tiktoken as an example
python - <<'PY'
import tiktoken, os

enc = tiktoken.get_encoding('gpt2')
with open(os.path.join(os.environ.get('DATA_DIR', 'dataset'), 'sample.txt')) as f:
    text = f.read()

tokens = enc.encode(text)
with open(os.path.join(os.environ.get('DATA_DIR', 'dataset'), 'sample.tok'), 'w') as f:
    f.write(' '.join(map(str, tokens)))
PY

echo "Dataset prepared at $DATA_DIR"
