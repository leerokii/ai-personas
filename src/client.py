"""Anthropic API wrapper with running cost tracking.

Wraps the Anthropic SDK so every call funnels through one place that:
  - loads the model, pricing, and budget cap from config.yaml
  - reads the API key from the environment (never from code or git)
  - accumulates input/output token usage and converts it to USD
  - refuses to exceed the configured budget cap

The unit price comes from config.yaml's `pricing` block so that changing the
model (and its rates) is a one-line config edit, not a code change.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

import anthropic

# Repo root = parent of this file's directory (src/).
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


class BudgetExceeded(RuntimeError):
    """Raised when a call would push total spend past the budget cap."""


@dataclass
class CostTracker:
    """Accumulates token usage and converts it to USD using config pricing."""

    input_per_mtok: float
    output_per_mtok: float
    budget_cap_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.n_calls += 1

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.input_per_mtok
            + self.output_tokens / 1_000_000 * self.output_per_mtok
        )

    def summary(self) -> str:
        return (
            f"calls={self.n_calls} "
            f"in={self.input_tokens:,}tok out={self.output_tokens:,}tok "
            f"cost=${self.cost_usd:.4f} / cap=${self.budget_cap_usd:.2f}"
        )


class Client:
    """Thin wrapper around anthropic.Anthropic with cost tracking + budget guard."""

    def __init__(self, config: dict | None = None):
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and paste "
                "your key, or export it in the environment."
            )

        self.config = config or load_config()
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        pricing = self.config["pricing"]
        self.tracker = CostTracker(
            input_per_mtok=pricing["input_per_mtok"],
            output_per_mtok=pricing["output_per_mtok"],
            budget_cap_usd=self.config["budget_cap_usd"],
        )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._lock = threading.Lock()  # guards the cost tracker for concurrent runs

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 16,
        temperature: float | None = None,
    ) -> str:
        """Send one user message, return the text of the first text block.

        Records usage on the cost tracker. Raises BudgetExceeded if the running
        cost has already passed the cap (checked before the call).
        """
        with self._lock:
            if self.tracker.cost_usd >= self.tracker.budget_cap_usd:
                raise BudgetExceeded(
                    f"Budget cap hit before call: {self.tracker.summary()}"
                )

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        with self._lock:
            self.tracker.record(
                response.usage.input_tokens, response.usage.output_tokens
            )

        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        return text

    def print_cost(self, label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}{self.tracker.summary()}")


if __name__ == "__main__":
    # Smoke check: construct the client and print the (zero) cost line.
    # Does not make an API call.
    c = Client()
    print(f"model={c.model} temperature={c.temperature}")
    c.print_cost("init")
