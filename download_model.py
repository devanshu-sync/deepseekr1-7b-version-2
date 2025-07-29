from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
local_path = "./deepseek_model"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(local_path)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(local_path)
