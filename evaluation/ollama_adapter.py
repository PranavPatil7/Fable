import requests, json

class OllamaAdapter:
    def __init__(self, model="granite3.3:2b", host="http://localhost:11434",
                 temperature=0.2, max_tokens=512):
        self.model = model
        self.url   = host.rstrip('/') + "/api/generate"
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
