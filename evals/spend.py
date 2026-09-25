"""Print OpenRouter spend so far for the key in OPENROUTER_API_KEY (never the key itself)."""

import json
import os
import sys

import httpx

key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    sys.exit("OPENROUTER_API_KEY is not set")
response = httpx.get(
    "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=20
)
data = response.json().get("data", {})
print(json.dumps({k: data.get(k) for k in ("usage", "usage_daily", "limit", "limit_remaining")}))
