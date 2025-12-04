from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch




# -----------------------------------
# CONFIG — CHANGE THESE FOR YOUR DATA
# -----------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # I used 0.5B cz smaller size in mac if you have heavy loading laptop you can go for 1.5B-Instruct
DATA_PATH = "data/train.jsonl"            # this is my dataset file
OUTPUT_DIR = "./qwen2_5_med_lora"         # this is where model will be saved this will generate automatically

MAX_LEN = 512  # i dont have GPU so i used 512 (if you have GPU you can use 1024 or if higher then 2056


# 1) Load dataset
print(" Loading dataset...")
dataset = load_dataset("json", data_files={"train": DATA_PATH})

# 2) Load tokenizer
print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# 3) Load base model in 4-bit (QLoRA memory-efficient mode)
print(" Loading base model (CPU, no 4-bit quantization)...")
# Running on macOS without CUDA: avoid bitsandbytes 4-bit quantization.
# Load model on CPU (or a CUDA-enabled machine if available and desired).
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)


# 4) Add LoRA adapters (train only small % of weights)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)


# 5) Format dataset into chat template for Qwen
def format_example(ex):
    system_prompt = (
        "You are a helpful assistant. "
        "For medical questions: always give safe guidance and say "
        "you are not a substitute for a doctor."
    )

    user_message = ex["instruction"]
    if ex.get("input") and ex["input"].strip():
        user_message += "\n\nContext:\n" + ex["input"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ex["output"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


print(" Formatting dataset...")
dataset = dataset.map(format_example)


# 6) Tokenize dataset
def tokenize_fn(ex):
    out = tokenizer(
        ex["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out


print(" Tokenizing dataset...")
tokenized = dataset["train"].map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 7) Data collator (for LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# 8) Training settings
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # keep 1 on MPS
    gradient_accumulation_steps=1,       # start small
    num_train_epochs=1,                  # just to test pipeline
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    bf16=False,                          # MPS uses float16, keep bf16=False
    optim="adamw_torch",
    report_to="none",
)
#you can change the upper values and see the affect on the model


# 9) Trainer
from transformers import Trainer
print("🚀 Starting Training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()

print("💾 Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training Finished! Model saved to:", OUTPUT_DIR)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
)
model.to(device)
