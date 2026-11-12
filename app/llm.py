import os
import requests
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


class GroqClient(LLMClient):
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.model = model

    def complete(self, system, user):
        r = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class MockLLM(LLMClient):
    """Scripted responses for tests."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.responses.pop(0)
