import runpod
# import time  
import torch
# import asyncio
# import threading
import os

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"


from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

# Load model and tokenizer outside the handler
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",  # Uses accelerate to infer device map
    # torch_dtype=torch.float16,  # Optional: can use float32 for more stability
    trust_remote_code=True)

# Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
model.eval()

# lock = threading.Lock()  # Prevent concurrent model access

def run_model(prompt):
    # with lock, torch.no_grad():
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return tokenizer.decode(outputs[0])
        
# async def handler(event):
def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  

    result = run_model(prompt)
    # loop = asyncio.get_event_loop()
    # result = await loop.run_in_executor(None, lambda: run_model(prompt))

    return result

# def update_request_rate(request_history):
#     # Collects real metrics about request patterns.
#     #
#     # Args:
#     #     request_history (list): A list of request timestamps
#     #
#     # Returns:
#     #     int: The new request rate

#     global request_rate
    
#     # Option 1: Track request count over a time window
#     current_time = time.time()
#     # Count requests in the last minute
#     recent_requests = [r for r in request_history if r > current_time - 60]
#     request_rate = len(recent_requests)
    
    # Option 2: Use an exponential moving average
    # request_rate = 0.9 * request_rate + 0.1 * new_requests
    
    # Option 3: Read from a shared metrics service like Redis
    # request_rate = redis_client.get('recent_request_rate')

# def adjust_concurrency(current_concurrency):
#     # Dynamically adjust the worker's concurrency level based on request load.
#     #
#     # Args:
#     #     current_concurrency (int): The current concurrency level
#     #
#     # Returns:
#     #     int: The new concurrency level

#     global request_rate
    
#     # In production, this would use real metrics
#     update_request_rate()
    
#     max_concurrency = 10  # Maximum allowable concurrency
#     min_concurrency = 1   # Minimum concurrency to maintain
#     high_request_rate_threshold = 50  # Threshold for high request volume
    
#     # Increase concurrency if under max limit and request rate is high
#     if (request_rate > high_request_rate_threshold and 
#         current_concurrency < max_concurrency):
#         return current_concurrency + 1
#     # Decrease concurrency if above min limit and request rate is low
#     elif (request_rate <= high_request_rate_threshold and 
#           current_concurrency > min_concurrency):
#         return current_concurrency - 1
    
#     return current_concurrency

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        # 'concurrency_modifier': adjust_concurrency 
    })
