FROM python:3.12-slim

WORKDIR /

# Set environment variables for HuggingFace cache
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV HF_DATASETS_CACHE=/workspace/hf_cache

# Create cache and model directories
RUN mkdir -p /workspace/hf_cache /workspace/model

# Copy and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace model and tokenizer
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B').save_pretrained('/workspace/model'); \
    AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B').save_pretrained('/workspace/model')"

# Copy your handler script
COPY main.py /main.py

# Start the container
CMD [\"python3\", \"-u\", \"main.py\"]
