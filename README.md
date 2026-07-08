# LLM From Scratch — GPT in Pure PyTorch

A complete, educational implementation of a GPT-style Large Language Model,
built from scratch in PyTorch. Structure and methodology follow Sebastian
Raschka's *Build a Large Language Model (From Scratch)*
(https://github.com/rasbt/LLMs-from-scratch), re-implemented as a clean,
modular Python package.

**Author:** Muhammad Titan Anugrah Santosa (titananugrah81@gmail.com)

## The six stages

| Stage | Module | What it does |
|---|---|---|
| 1. Data & tokenization | `llm/tokenizer.py`, `llm/dataset.py` | Byte-pair encoding (trainable from scratch, or GPT-2 BPE via `tiktoken`), sliding-window dataset, token + positional embeddings |
| 2. Attention | `llm/attention.py` | Causal (masked) multi-head self-attention with dropout |
| 3. GPT architecture | `llm/model.py` | LayerNorm, GELU, feed-forward, transformer blocks, full GPT model (124M-param GPT-2 layout by default) |
| 4. Pretraining | `llm/train.py`, `llm/generate.py` | Next-token cross-entropy training loop, evaluation, checkpointing, temperature/top-k text generation |
| 5. Classification finetuning | `llm/finetune_classification.py` | Swap the output head and finetune the pretrained model as a binary (spam) classifier |
| 6. Instruction finetuning | `llm/finetune_instruction.py` | Alpaca-style prompt formatting, loss masking with ignore_index=-100, supervised instruction finetuning |

## Setup

```bash
pip install -r requirements.txt
```

`tiktoken` is optional — if absent, the built-in trainable BPE tokenizer is used.

## Quick start

```bash
# Stage 4: pretrain a small GPT on any plain-text file
python -m llm.train --data data/sample.txt --context 256 --emb 384 \
    --layers 6 --heads 6 --epochs 10

# Generate text from the trained checkpoint
python -m llm.generate --checkpoint checkpoints/gpt_pretrained.pt \
    --prompt "Every effort moves you" --max-new-tokens 50 --temperature 0.8 --top-k 40

# Stage 5: finetune as a spam classifier (CSV with columns: label,text)
python -m llm.finetune_classification --data data/spam.csv \
    --checkpoint checkpoints/gpt_pretrained.pt

# Stage 6: instruction finetuning (JSON list of {instruction, input, output})
python -m llm.finetune_instruction --data data/instructions.json \
    --checkpoint checkpoints/gpt_pretrained.pt
```

## Smoke test (no data needed)

```bash
python -m llm.smoke_test
```

Runs a forward pass, a generation step, and a single training step on random
tokens with a tiny config, verifying every module end to end.

## Repository layout

```
llm-from-scratch/
├── README.md
├── requirements.txt
├── data/sample.txt              # small public-domain text for quick experiments
└── llm/
    ├── __init__.py
    ├── config.py                # GPTConfig dataclass + presets (gpt2-small ... gpt2-xl)
    ├── tokenizer.py             # trainable BPE + optional tiktoken GPT-2 BPE
    ├── dataset.py               # sliding-window next-token dataset & dataloader
    ├── attention.py             # causal multi-head self-attention
    ├── model.py                 # transformer blocks + GPTModel
    ├── generate.py              # sampling (greedy / temperature / top-k)
    ├── train.py                 # pretraining loop
    ├── finetune_classification.py
    ├── finetune_instruction.py
    └── smoke_test.py
```

## References

- S. Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024.
- A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- A. Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019.
