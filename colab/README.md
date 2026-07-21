# GEPA Evolution Notebook

This directory contains a self-contained Google Colab notebook for running **GEPA (Genetic Pareto Algorithm) Prompt Evolution** on any category using free Colab GPU (Tesla T4, 16GB VRAM).

## Files

| File | Description |
|------|-------------|
| `GEPA_Evolution_Notebook.ipynb` | The main Colab notebook |
| `README.md` | This file |

## Quick Start

1. **Open the notebook in Colab:**
   - Upload to Google Drive and open with Colab, OR
   - Go to [https://colab.research.google.com/](https://colab.research.google.com/) → File → Upload Notebook

2. **Enable GPU:**
   - Runtime → Change runtime type → Hardware accelerator → **T4 GPU**

3. **Run cells in order** (Cells 1 → 7):
   - **Cell 1**: Installs dependencies (llama-cpp-python with CUDA, vaderSentiment, tqdm, huggingface_hub), mounts Google Drive, creates output directories
   - **Cell 2**: Downloads a GGUF model (choose from Qwen 2.5 1.5B, Qwen Coder 1.5B, or Gemma 3 1B)
   - **Cell 3**: Upload your eval dataset (JSON file, URL, or use the demo sentiment dataset)
   - **Cell 4**: Loads the GEPA engine (no user action needed)
   - **Cell 5**: Configure category, model, and evolution parameters
   - **Cell 6**: **Run the evolution** — creates seed prompts, evaluates against your data, evolves across generations
   - **Cell 7**: View results, Pareto front, and download final report

## Eval Data Format

The notebook expects a JSON file with this structure:

```json
[
  {
    "category": "sentiment",
    "prompt": "I love this product!",
    "expected_answer": "positive"
  },
  {
    "category": "ner",
    "prompt": "John works at Google in New York.",
    "expected_answer": "[\"John\", \"Google\", \"New York\"]"
  }
]
```

### Fields
- **`category`** (string): The classification/analysis category
- **`prompt`** (string): The input text to classify
- **`expected_answer`** (string): The correct answer (used for scoring)

### Category Examples
- `sentiment` → expected: `positive` / `negative` / `neutral`
- `ner` → expected: entity names
- `factual` → expected: factual labels
- `toxicity` → expected: `toxic` / `non-toxic`
- `emotion` → expected: `joy` / `sadness` / `anger` / etc.
- `intent` → expected: intent category

## Configuration

In **Cell 5**, you can edit:

```python
CATEGORY = "sentiment"           # Auto-detected, override if needed
MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"  # Downloaded model filename
GENERATIONS = 3                  # Number of evolution generations
POPULATION_SIZE = 8              # Cells per generation
CELLS_PER_MODEL = 4              # Cells evaluated per model load cycle
```

## How GEPA Works

1. **Seed Generation**: Creates initial prompt templates from base patterns
2. **Evaluation**: Runs each prompt against all eval questions using the GGUF model
3. **Fuzzy Scoring**: 4-cascade matching:
   - Exact match → substring match → numeric 1% tolerance → token overlap ≥ 80%
4. **Selection**: Top 50% accuracy cells become elites
5. **Mutation**: Creates offspring via synonym swap, constraint addition, template swap, temperature change
6. **Pareto Tracking**: Monitors (accuracy, token count, latency) trade-offs across all generations

## Output

Results are saved to Google Drive at:
```
/MyDrive/GEPA_Results/
├── models/             # Downloaded GGUF models (cached)
├── snapshots/          # Per-generation snapshots
│   ├── gen_0.json
│   ├── gen_1.json
│   └── ...
├── reports/
│   ├── final_report.json   # Complete results
│   └── summary.txt         # Text summary
└── eval_data.json          # Your uploaded eval dataset
```

## VRAM Management

The notebook is designed for Colab's free T4 GPU (16GB VRAM, ~8GB usable for models):
- Models are **loaded** → **evaluate several cells** → **unloaded** to free VRAM
- `CELLS_PER_MODEL` controls how many cells are evaluated per load cycle
- If you hit CUDA OOM errors, reduce `POPULATION_SIZE` or `CELLS_PER_MODEL`
- The smallest model (Gemma 3 1B, ~0.78 GB) is the safest choice

## Requirements

- Google account (for Colab + Drive)
- No local setup needed — everything runs in the browser
- Internet connection for initial dependency installation and model download
