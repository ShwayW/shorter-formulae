"""LLM backends for the one-shot searcher.

Two thin, *synchronous* wrappers exposing a single `complete(system, user) -> str`:

  OpenAICompatLLM -- any OpenAI-compatible /v1/chat/completions server. Covers the
                     self-hosted paths we use for free: `vllm serve <model>` on a
                     GPU node (http://localhost:8000/v1) and ollama on a workstation
                     (http://localhost:11434/v1). Also works with the paid clouds.
  GeminiLLM       -- Vertex/GenAI, mirroring llm_methods/llmsr/sampler.py:GeminiLLM so the
                     one-shot and LLMSR numbers come off the same backend.

Deliberately synchronous and un-batched: a one-shot run makes one call per problem,
so there is nothing to overlap, and the simpler code path makes the timing honest.
"""

from __future__ import annotations


class OneShotLLM:
    """Interface: one chat completion, text in / text out."""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAICompatLLM(OneShotLLM):
    def __init__(self, api_model: str, api_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", temperature: float = 0.0,
                 max_tokens: int = 1024, timeout: float = 300.0) -> None:
        from openai import OpenAI
        self.api_model = api_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=api_url, api_key=api_key, timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content or ""


class GeminiLLM(OneShotLLM):
    def __init__(self, api_model: str, project: str | None = None,
                 location: str = "global", vertexai: bool = True,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 thinking_budget: int | None = None) -> None:
        from google import genai
        self.api_model = api_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.client = (genai.Client(vertexai=True, project=project, location=location)
                       if vertexai else genai.Client())

    def complete(self, system: str, user: str) -> str:
        from google.genai import types
        cfg = dict(system_instruction=system,
                   temperature=self.temperature,
                   max_output_tokens=self.max_tokens)
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        resp = self.client.models.generate_content(
            model=self.api_model, contents=user,
            config=types.GenerateContentConfig(**cfg))
        # Reasoning models split the answer across parts; concatenate the non-thought text.
        out = []
        for cand in (resp.candidates or []):
            for part in (getattr(cand.content, "parts", None) or []):
                if getattr(part, "thought", False):
                    continue
                if getattr(part, "text", None):
                    out.append(part.text)
        return "\n".join(out) if out else (resp.text or "")


def build_llm(cfg: dict) -> OneShotLLM:
    """Construct the backend named by cfg['api_type'] from a yaml config dict."""
    api_type = cfg.get("api_type", "vllm")
    if api_type in ("vllm", "ollama", "local", "openai"):
        import os
        # $ONESHOT_API_URL lets an sbatch wrapper point the run at this job's own
        # vLLM port (same trick as eval_llmsr.py's $LLMSR_API_URL).
        api_url = os.environ.get("ONESHOT_API_URL", cfg.get("api_url", "http://localhost:8000/v1"))
        return OpenAICompatLLM(api_model=cfg["api_model"], api_url=api_url,
                               api_key=cfg.get("api_key", "EMPTY"),
                               temperature=cfg.get("temperature", 0.0),
                               max_tokens=cfg.get("max_tokens", 1024),
                               timeout=cfg.get("api_timeout", 300.0))
    if api_type == "gemini":
        return GeminiLLM(api_model=cfg["api_model"], project=cfg.get("project"),
                         location=cfg.get("location", "global"),
                         vertexai=cfg.get("vertexai", True),
                         temperature=cfg.get("temperature", 0.0),
                         max_tokens=cfg.get("max_tokens", 1024),
                         thinking_budget=cfg.get("thinking_budget", None))
    raise ValueError(f"unknown api_type={api_type!r}; expected vllm/ollama/local/openai/gemini")
