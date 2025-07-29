
```bash
 curl -X POST https://api.runpod.ai/v2/yf1yw4qkstuk5n/runsync \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer Token' \
    -d '{"input":{"prompt":"Hello World"}}'
```

```json
{
  "input": {
    "prompt": "Your input here"
  },
  "policy": {
    "executionTimeout": 10000,  // 10 seconds in milliseconds
    "lowPriority": true,    //When true, job won’t trigger worker scaling
    "ttl": 3600000  // 1 hour in milliseconds
  }
}
```

rate limit

```text
/runsync	POST	2000 requests per 10 seconds	400 concurrent
```

test locally

```bash
python main.py \
  --rp_serve_api \
  --rp_api_port 8080 \
  --rp_api_concurrency 4
```