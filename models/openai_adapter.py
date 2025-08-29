
"""Adapter for querying OpenAI ChatCompletion models (e.g., GPT‑4, GPT‑3.5).

Requires environment variable OPENAI_API_KEY.
Usage:
    from models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    response = adapter.generate(prompt)
"""
import os, openai, time, backoff

class OpenAIAdapter:
    def __init__(self, model="gpt-4o-mini", temperature=0.2, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")

    @backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=5)
    def generate(self, prompt:str)->str:
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return resp.choices[0].message.content.strip()
