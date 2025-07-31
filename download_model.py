from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
local_path = "./deepseek_model"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(local_path)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(local_path)
