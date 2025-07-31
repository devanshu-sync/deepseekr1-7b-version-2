import runpod
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Change Hugging Face cache directory
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

# Model name
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"  # ✅ DeepSeek Coder model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ✅ Load model with INT8 quantization using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # Automatically map layers to available GPUs
    load_in_8bit=True,        # ✅ Enable INT8 quantization
    trust_remote_code=True
)

model.eval()

def run_model(prompt):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return tokenizer.decode(outputs[0])  # ✅ Raw output, no cleaning

def handler(event):
    print("Worker Start")
    user_input = event.get("input", {})
    prompt = user_input.get("prompt")

    if not prompt:
        return {"status": "error", "message": "Missing 'prompt' in input"}

    try:
        generated_text = run_model(prompt)
        return {  # ✅ Return raw text
            "status": "success",
            "input_prompt": prompt,
            "generated_text": generated_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
