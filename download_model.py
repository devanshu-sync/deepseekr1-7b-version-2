from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
local_path = "./deepseek_model_int8"

# Create directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

# ✅ Download and save tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(local_path)
print(f"Tokenizer saved at {local_path}")

# ✅ Load model with INT8 quantization
print("Downloading and loading model with INT8 quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",       # Uses accelerate to map model to GPU
    load_in_8bit=True        # ✅ Enable INT8 quantization
)

# ✅ Save quantized model
# Note: Quantized models can't be saved in standard format easily.
# We use safe_serialization=False for compatibility with bitsandbytes
print("Saving quantized model...")
model.save_pretrained(local_path, safe_serialization=False)
print(f"INT8 quantized model saved at {local_path}")
