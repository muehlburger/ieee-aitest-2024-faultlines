# FaultLines

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23890440.svg)](https://doi.org/10.5281/zenodo.23890440)

Code, data, and results for the paper *FaultLines - Evaluating the Efficacy of Open-Source Large Language Models for Fault Detection in Cyber-Physical Systems* (H. Mühlburger and F. Wotawa, IEEE AITest 2024, pp. 47-54, [doi:10.1109/AITest62860.2024.00014](https://doi.org/10.1109/AITest62860.2024.00014)).

Open-weight language models judge sensor rows of a simulated battery system. Each test row is compared with the ten most similar normal rows retrieved from a vector store. No model is trained or fine-tuned, and no labeled faults are used.

## Repository layout

| Path | Content |
|---|---|
| `code/0-text-encode.py` | Encodes the sensor rows as text |
| `code/1-load-vectorstore.py` | Embeds the normal rows and stores them in Qdrant |
| `code/2-detect-anomalies.py` | Runs one model over a range of test rows |
| `code/3-evaluate.py` | Computes F1, precision, and recall from `results/` |
| `code/run.sh` | Runs all experiments of the paper |
| `data/` | Raw battery signals and their text encodings |
| `results/` | Model answers of the published runs |

## Setup

Python 3.10 was used. Install the pinned dependencies, start Qdrant, and install [Ollama](https://ollama.com):

```bash
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant
```

The published numbers can be recomputed from `results/` without Qdrant or Ollama:

```bash
python code/3-evaluate.py
```

To repeat the experiments, pull the models with `ollama pull <tag>` and run `code/run.sh`. `2-detect-anomalies.py` takes `--host`, `--port`, and `--device` for other setups. It defaults to Qdrant on `localhost` and to the CPU; `run.sh` uses these defaults.

## Notes on reproduction

- **Model versions:** The paper used Ollama v0.1.30 with the tags available in March 2024. Tags such as `mistral` have moved since then, so a new run can return different answers. The answers of the published runs are in `results/`.
- **Label boundary:** The fault starts at row index 4001 (`time_s` = 2000.5). The 10,801-sample evaluation uses this boundary. The 200-sample evaluation of the paper labels the last 100 rows as faulty, starting at index 4000, so one normal row counts as faulty. `3-evaluate.py` keeps this boundary to reproduce the published values. With the correct boundary, F1 changes by at most 0.007 for any model.
- **Data:** The files named `roundend` keep the spelling of the original files. `0-text-encode.py` reproduces the encoded files in `data/` byte for byte from the raw CSVs.

## Citation

```bibtex
@inproceedings{muehlburger2024faultlines,
  author    = {M{\"u}hlburger, Herbert and Wotawa, Franz},
  title     = {{FaultLines} - Evaluating the Efficacy of Open-Source Large Language Models for Fault Detection in Cyber-Physical Systems},
  booktitle = {2024 IEEE International Conference on Artificial Intelligence Testing (AITest)},
  year      = {2024},
  pages     = {47--54},
  doi       = {10.1109/AITest62860.2024.00014}
}
```

## License

[Apache 2.0](LICENSE)
