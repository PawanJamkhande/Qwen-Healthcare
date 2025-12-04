from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # same as training
ADAPTER_PATH = "./qwen2_5_med_lora"        # path where LoRA saved

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

print("🔌 Loading tokenizer & base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype
).to(device)

print("🔌 Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.to(device)
model.eval()

def chat(message):
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": message}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


print("\n🤖 Model ready! Ask a medical question.\n")

while True:
    user_input = input("🧑‍💻 You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Bye!")
        break
    
    answer = chat(user_input)
    print("💡 AI:", answer, "\n")
