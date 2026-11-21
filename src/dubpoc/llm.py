"""Çeviri için LLM istemcileri: Anthropic (varsayılan) ve OpenAI-uyumlu uç (örn. yerel Ollama).

Her iki yol da JSON şemasıyla kısıtlanmış çıktı ister; dönen JSON kodda doğrulanır.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            a, b = text.find(open_c), text.rfind(close_c)
            if a != -1 and b > a:
                try:
                    return json.loads(text[a: b + 1])
                except json.JSONDecodeError:
                    continue
    raise LLMError(f"LLM geçerli JSON döndürmedi: {text[:500]}")


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg.get("provider", "anthropic")
        self.model = cfg["model"]
        self._client = None

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"

    def json_call(self, system: str, user: str, schema: dict, name: str = "result") -> Any:
        if self.provider == "anthropic":
            return self._anthropic(system, user, schema)
        if self.provider == "openai_compat":
            return self._openai_compat(system, user, schema, name)
        raise LLMError(f"Bilinmeyen translate.provider: {self.provider}")

    # ---- Anthropic -------------------------------------------------------------------------
    def _anthropic(self, system: str, user: str, schema: dict) -> Any:
        import anthropic

        if self._client is None:
            key = os.environ.get(self.cfg.get("api_key_env") or "ANTHROPIC_API_KEY")
            # anahtar yoksa SDK; ANTHROPIC_AUTH_TOKEN / `ant auth login` profilini dener
            self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema},
                           "effort": self.cfg.get("effort", "medium")},
        )
        try:
            if self.cfg.get("fallbacks", True):
                # Güvenlik sınıflandırıcısı isteği reddederse sunucu tarafında yedek modele yönlendir
                resp = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic kimlik doğrulaması başarısız. ANTHROPIC_API_KEY ayarlayın ya da "
                           "config'te translate.provider: openai_compat (Ollama) kullanın.") from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"Anthropic hız sınırı: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API hatası {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Anthropic bağlantı hatası: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("Model isteği reddetti (refusal).")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Yanıt max_tokens sınırında kesildi; sahneyi bölün.")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        return _extract_json(text)

    # ---- OpenAI-uyumlu (Ollama vb.) ----------------------------------------------------------
    def _openai_compat(self, system: str, user: str, schema: dict, name: str) -> Any:
        base = (self.cfg.get("base_url") or "http://localhost:11434/v1").rstrip("/")
        headers = {}
        key = os.environ.get(self.cfg.get("api_key_env") or "", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.3,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": name, "schema": schema, "strict": True}},
            **(self.cfg.get("extra_body") or {}),
        }
        try:
            r = httpx.post(f"{base}/chat/completions", json=body, headers=headers, timeout=600)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"openai_compat isteği başarısız ({base}): {e}") from e
        msg = r.json()["choices"][0]["message"]
        return _extract_json(msg.get("content") or "")

    def release(self) -> None:
        """Yerel Ollama modelini VRAM'den boşalt (sonraki GPU worker'ları 6 GB'a sığsın)."""
        if self.provider != "openai_compat":
            return
        base = (self.cfg.get("base_url") or "").rstrip("/")
        if ":11434" not in base:
            return
        root = base[:-3] if base.endswith("/v1") else base
        try:
            httpx.post(f"{root}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except httpx.HTTPError:
            pass
