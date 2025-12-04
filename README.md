# Qwen2.5 Medical Fine-tuning

A small experiment to fine-tune Qwen2.5-0.5B-Instruct on synthetic medical safety data using LoRA on macOS.

## Files
- `train_qwen.py` — fine-tuning script
- `infer_qwen.py` — test the fine-tuned model
- `data/train.jsonl` — synthetic training dataset (not uploaded)
- `data/val.jsonl` — synthetic validation dataset (not uploaded)

# Qwen Healthcare – Fine-Tuning Qwen2.5 for Medical Q&A 

This repo contains a **minimal end-to-end setup** for fine-tuning [Qwen2.5 Instruct](https://huggingface.co/Qwen) models on **healthcare / medical-style text** using **LoRA** on a local machine (e.g. macOS with MPS).

>  **Disclaimer**  
> This project is for **research & educational purposes only**.  
> The resulting model is **not** a medical device, not validated clinically, and **must not be used for real diagnosis or treatment.**  
> Always consult qualified healthcare professionals.

---

## What this repo includes

Core components:

- `train_qwen.py` – fine-tunes a Qwen2.5-Instruct model using:
  - [Transformers](https://github.com/huggingface/transformers)
  - [Datasets](https://github.com/huggingface/datasets)
  - [PEFT / LoRA](https://github.com/huggingface/peft) (parameter-efficient fine-tuning)
- `infer_qwen.py` – simple CLI script to **chat** with the fine-tuned model
- `requirements.txt` – minimal Python dependencies
- `.gitignore` – protects:
  - Local datasets
  - Model weights / checkpoints
  - HF cache & large binary files

The training script is set up to work **locally** (e.g. on a Mac) without CUDA and without `bitsandbytes` 4-bit quantization.


## Tech Stack

- **Base Model:** Qwen2.5 Instruct (configurable, usually:  
  `Qwen/Qwen2.5-0.5B-Instruct` for local experiments)
- **Frameworks & Libraries:**
  - `transformers` – model + tokenizer + Trainer
  - `datasets` – JSONL loading & processing
  - `peft` – LoRA fine-tuning
  - `torch` – backend (CPU / MPS / GPU)
- **Hardware target:**  
  - macOS (Apple Silicon with MPS) or CPU-only machines  
  - For serious training → cloud GPU recommended

---

## Repository Structure

```bash
Qwen-Healthcare/
├── train_qwen.py        # fine-tuning script (LoRA + Qwen2.5)
├── infer_qwen.py        # inference / chat CLI for the fine-tuned model
├── requirements.txt     # python deps
├── README.md            # this file
└── .gitignore           # ignores data, weights, caches, etc.
```

## The repo uses .gitignore to avoid commiting for:

Data privacy: In real setups, medical or sensitive data must not be public.
Repository size: Model weights & .safetensors are huge and not suitable for normal Git.
Reproducibility: Anyone can regenerate weights by:
Downloading the base model from Hugging Face
Running train_qwen.py on their own dataset
You’re expected to create the data/ folder and model output folder locally, not pull them from GitHub.

## Dataset Format (train.jsonl / val.jsonl)

The scripts expect JSONL files with each line like:
{"instruction": "What is hypertension?", "input": "", "output": "Hypertension means... Please consult a doctor."}

## Fields:

instruction: user question / task
input: optional extra context (can be "")
output: desired answer (what you want the model to learn)
Example minimal data/train.jsonl:
