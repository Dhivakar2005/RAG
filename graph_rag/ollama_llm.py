import requests
import json
from config import OLLAMA_URL, MODEL_NAME

def generate_completion(prompt, format_json=False):
    """
    Sends a request to the local Ollama REST API.
    Uses the phi3:mini model as defined in config.py
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    
    if format_json:
        payload["format"] = "json"
        
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""
