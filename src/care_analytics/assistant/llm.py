"""Chat model clients behind one small interface.

Two providers:

* ``ollama`` (default): a local model through Ollama's native API, with "thinking" turned off.
  No customer data leaves the machine.
* ``deepseek``: DeepSeek's OpenAI-compatible API, configured by ``DEEPSEEK_*`` variables.
* ``openai``: any OpenAI-compatible endpoint (OpenAI, Azure OpenAI v1 endpoint ...).

Environment: ``ASSISTANT_PROVIDER``, ``ASSISTANT_MODEL``, ``ASSISTANT_BASE_URL``,
``ASSISTANT_API_KEY``; for DeepSeek ``DEEPSEEK_BASE_URL``, ``DEEPSEEK_API_KEY``,
``DEEPSEEK_MODEL``, ``DEEPSEEK_TEMPERATURE``, ``DEEPSEEK_TIMEOUT_SECONDS``,
``DEEPSEEK_MAX_RETRIES``. A ``.env`` file in the working directory is read if present;
variables already set in the environment win.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides the real environment."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached_prompt_tokens: int = 0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict]) -> Reply: ...


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Drop the reasoning part of a reply, also when the template omits the opening tag."""
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


@dataclass
class OpenAICompatible:
    base_url: str = field(
        default_factory=lambda: os.environ.get("ASSISTANT_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(default_factory=lambda: os.environ.get("ASSISTANT_MODEL", "gpt-4.1-mini"))
    api_key: str = field(default_factory=lambda: os.environ.get("ASSISTANT_API_KEY", ""))
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2
    # model-specific request fields, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    extra_body: dict | None = field(
        default_factory=lambda: json.loads(os.environ.get("ASSISTANT_EXTRA_BODY", "null"))
    )
    max_tokens: int | None = field(
        default_factory=lambda: (
            int(os.environ["ASSISTANT_MAX_TOKENS"])
            if os.environ.get("ASSISTANT_MAX_TOKENS")
            else None
        )
    )

    @property
    def name(self) -> str:
        return self.model

    def chat(self, messages: list[dict]) -> Reply:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout,
                        max_retries=self.max_retries)  # fmt: skip
        t0 = time.perf_counter()
        kw = {"extra_body": self.extra_body} if self.extra_body else {}
        if self.max_tokens:
            kw["max_tokens"] = self.max_tokens
        r = client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, **kw
        )
        text = strip_reasoning(r.choices[0].message.content or "")
        u = r.usage
        return Reply(
            text=text,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            seconds=time.perf_counter() - t0,
            # DeepSeek reports prompt-cache hits; they are billed at a fraction of the price
            cached_prompt_tokens=getattr(u, "prompt_cache_hit_tokens", 0) or 0,
        )


def deepseek_model() -> OpenAICompatible:
    env = os.environ
    return OpenAICompatible(
        base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=env.get("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=env.get("DEEPSEEK_API_KEY", ""),
        temperature=float(env.get("DEEPSEEK_TEMPERATURE", "0")),
        timeout=float(env.get("DEEPSEEK_TIMEOUT_SECONDS", "60")),
        max_retries=int(env.get("DEEPSEEK_MAX_RETRIES", "2")),
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
            text=strip_reasoning(d["message"]["content"]),
            prompt_tokens=d.get("prompt_eval_count", 0),
            completion_tokens=d.get("eval_count", 0),
            seconds=time.perf_counter() - t0,
        )


def default_model(provider: str | None = None) -> ChatModel:
    load_dotenv()
    provider = (provider or os.environ.get("ASSISTANT_PROVIDER", "ollama")).lower()
    if provider == "deepseek":
        return deepseek_model()
    if provider == "openai":
        return OpenAICompatible()
    return OllamaChat()


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
