# LLM from Scratch

LLM from Scratch adalah implementasi mini GPT-style language model menggunakan PyTorch. Project ini dibuat untuk mempelajari komponen utama large language model secara langsung: tokenization, dataset windowing, causal self-attention, transformer block, training loop, dan text generation.

Model di repository ini sengaja dibuat kecil agar mudah dijalankan di CPU dan cocok untuk eksperimen edukatif. Fokus utama project adalah memahami mekanisme dasar LLM, bukan menghasilkan model produksi berskala besar.

## Features

- GPT-style transformer decoder.
- Causal multi-head self-attention.
- Token embedding dan positional embedding.
- Feed-forward network dengan GELU.
- Layer normalization dan residual connection.
- Sliding-window dataset untuk next-token prediction.
- Training CLI sederhana.
- Text generation CLI dengan greedy decoding, temperature, dan top-k sampling.
- Unit tests untuk tokenizer, dataset, model forward pass, dan generation.
- Fallback byte-level tokenizer agar demo tetap berjalan saat encoding GPT-2 dari `tiktoken` belum tersedia offline.

## Project Structure

```text
.
|-- configs/                 # Experiment configuration
|-- data/
|   |-- sample/              # Small demo corpus
|   `-- README.md
|-- notebooks/               # Exploratory notebooks
|-- src/
|   `-- llm_from_scratch/
|       |-- dataset.py       # Dataset and dataloader utilities
|       |-- generate.py      # Text generation CLI and helper
|       |-- model.py         # GPT model implementation
|       |-- tokenizer.py     # Tokenizer utilities
|       `-- train.py         # Training CLI
|-- tests/                   # Unit tests
|-- pyproject.toml
`-- requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development dependencies:

```bash
python -m pip install -e .[dev]
```

## Run Tests

```bash
pytest
```

Expected result:

```text
5 passed
```

## Generate Text

The generation command initializes a tiny untrained GPT model and generates tokens from a prompt:

```bash
python -m llm_from_scratch.generate --prompt "Every effort" --max-new-tokens 8
```

Optional sampling parameters:

```bash
python -m llm_from_scratch.generate --prompt "Every effort" --max-new-tokens 20 --temperature 0.8 --top-k 20
```

Because the default model is untrained, generated text is expected to be noisy. The command is primarily a smoke test for the model and decoding pipeline.

## Train a Tiny Model

Run a minimal CPU-friendly training demo:

```bash
python -m llm_from_scratch.train --data-path data/sample/tiny_corpus.txt --epochs 1 --context-length 8 --stride 4
```

The sample corpus is intentionally small, so the result is only useful for validating the training loop. For meaningful language modeling experiments, use a larger public-domain dataset and increase model size gradually.

## Core Concepts Implemented

### Tokenization

The project uses `tiktoken` GPT-2 tokenization when available. If the encoding files are not available locally and the environment cannot download them, the code falls back to a simple byte-level tokenizer.

### Dataset Windowing

Text is converted into token IDs, then split into input-target pairs:

- Input: token sequence from position `i` to `i + context_length`
- Target: same sequence shifted one token to the right

This trains the model for next-token prediction.

### Transformer Decoder

The model follows the GPT decoder pattern:

- Token embedding
- Positional embedding
- Repeated transformer blocks
- Final layer normalization
- Linear language modeling head

Each transformer block contains causal multi-head attention, feed-forward layers, dropout, residual connections, and layer normalization.

## Dataset Notes

Use datasets with clear redistribution rights. Public-domain text or permissively licensed corpora are recommended for portfolio publication.

Large local datasets should be stored under:

```text
data/raw/
data/processed/
```

These folders are ignored by Git by default.

## Limitations

- The default model configuration is intentionally small.
- No checkpoint saving/loading is included yet.
- No validation split or evaluation metrics beyond training loss are included yet.
- Notebook outputs may need cleanup before publishing.
- This project is for learning and experimentation, not production language modeling.

## Roadmap

- Add checkpoint save/load support.
- Add YAML config loading for training.
- Add validation loss and perplexity reporting.
- Add public-domain dataset download script.
- Add CI workflow for automated tests.
