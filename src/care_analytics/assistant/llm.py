"""Chat model clients behind one small interface.

Two providers:

* ``ollama`` (default): a local model through Ollama's native API, with "thinking" turned off.
  No customer data leaves the machine.
* ``openai``: any OpenAI-compatible endpoint (OpenAI, Azure OpenAI v1 endpoint, DeepSeek ...).

Environment: ``ASSISTANT_PROVIDER``, ``ASSISTANT_MODEL``, ``ASSISTANT_BASE_URL``,
``ASSISTANT_API_KEY`` (openai only).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict]) -> Reply: ...


_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class OpenAICompatible:
    base_url: str = field(
        default_factory=lambda: os.environ.get("ASSISTANT_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(default_factory=lambda: os.environ.get("ASSISTANT_MODEL", "gpt-4.1-mini"))
    api_key: str = field(default_factory=lambda: os.environ.get("ASSISTANT_API_KEY", ""))
    temperature: float = 0.0
    timeout: float = 120.0

    @property
    def name(self) -> str:
        return self.model

    def chat(self, messages: list[dict]) -> Reply:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        text = _THINK.sub("", r.choices[0].message.content or "").strip()
        u = r.usage
        return Reply(
            text=text,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            seconds=time.perf_counter() - t0,
        )


@dataclass
class OllamaChat:
    model: str = field(default_factory=lambda: os.environ.get("ASSISTANT_MODEL", "qwen3.5-32k:2b"))
    base_url: str = field(
        default_factory=lambda: os.environ.get("ASSISTANT_BASE_URL", "http://localhost:11434")
    )
    num_ctx: int = 8192
    max_tokens: int = 512  # a query never needs more; stops a model that keeps going
    timeout: float = 600.0

    @property
    def name(self) -> str:
        return self.model

    def chat(self, messages: list[dict]) -> Reply:
        body = {
            "model": self.model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
        }  # fmt: skip
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )  # fmt: skip
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - configured URL
            d = json.loads(r.read())
        return Reply(
            text=_THINK.sub("", d["message"]["content"]).strip(),
            prompt_tokens=d.get("prompt_eval_count", 0),
            completion_tokens=d.get("eval_count", 0),
            seconds=time.perf_counter() - t0,
        )


def default_model() -> ChatModel:
    provider = os.environ.get("ASSISTANT_PROVIDER", "ollama").lower()
    return OpenAICompatible() if provider == "openai" else OllamaChat()


@dataclass
class ScriptedModel:
    """Offline stand-in for tests: replies are looked up from the user's question."""

    replies: dict[str, list[str]]
    name: str = "scripted"
    calls: list[list[dict]] = field(default_factory=list)

    def chat(self, messages: list[dict]) -> Reply:
        self.calls.append(list(messages))  # a copy: the caller keeps appending
        question = next(m["content"] for m in messages if m["role"] == "user")
        queue = self.replies.get(question, ["REFUSE: no scripted reply"])
        n_before = sum(1 for m in messages if m["role"] == "assistant")
        return Reply(text=queue[min(n_before, len(queue) - 1)])
