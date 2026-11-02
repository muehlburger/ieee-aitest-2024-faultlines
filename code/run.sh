#!/bin/sh
# Reproduces the runs of the paper. Requires a running Qdrant and Ollama.
set -e
cd "$(dirname "$0")"

python 1-load-vectorstore.py

for model in mistral gemma:instruct; do
    python 2-detect-anomalies.py -s 0 -e 10801 -m "$model"
done

for model in mistral gemma:instruct mixtral command-r codellama qwen:14b mistral-openorca dolphin-mixtral; do
    python 2-detect-anomalies.py -s 3900 -e 4100 -m "$model"
done

python 3-evaluate.py
