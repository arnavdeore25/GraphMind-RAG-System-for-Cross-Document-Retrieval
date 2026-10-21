"""Single shared Gemini client (google-genai SDK) with retry + JSON helper."""
import json
import re
import time

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, LLM_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)


def generate(prompt: str, json_mode: bool = False, retries: int = 4) -> str:
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain",
        temperature=0,
    )
    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=LLM_MODEL, contents=prompt, config=cfg)
            return (r.text or "").strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # handles rate limits


def generate_json(prompt: str) -> dict:
    txt = generate(prompt, json_mode=True)
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {}
