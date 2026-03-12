"""Domain plugin ABC + registry.

A domain plugin connects nanoversight to a specific system. Two required
methods (observe, format_context) and two optional (parse_recommendation, act).
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Domain(ABC):
    """Base class for nanoversight domain plugins."""

    # Unique identifier — used in CLI --domain flag
    DOMAIN_ID: str = ""

    # Display name for logs
    DOMAIN_NAME: str = ""

    # Seed questions — asked on first run before grow() generates new ones
    SEED_QUESTIONS: list[str] = []

    @abstractmethod
    def observe(self) -> dict[str, Any]:
        """Collect current metrics from the system being watched.

        Returns a flat dict of metric_name → value. Values should be
        numeric or short strings. Called every cycle.
        """
        ...

    @abstractmethod
    def format_context(
        self,
        metrics: dict[str, Any],
        conclusions: list[dict],
        changes: list[str],
    ) -> str:
        """Build a context string for the LLM.

        Args:
            metrics: Current observe() output.
            conclusions: List of settled conclusions (dicts with
                        'pattern', 'conclusion', 'confidence' keys).
            changes: List of human-readable change descriptions.

        Returns:
            A string that will be prepended to the LLM prompt.
        """
        ...

    def parse_recommendation(self, llm_output: str) -> dict | None:
        """Extract a structured recommendation from LLM output.

        Return None if no actionable recommendation found.
        Default: no recommendations (advisory only).
        """
        return None

    def act(self, recommendation: dict) -> bool:
        """Execute a recommendation. Return True if acted on.

        Default: advisory only — never acts.
        """
        return False


# ── Registry ─────────────────────────────────────────────────────────────────

_registry: dict[str, type[Domain]] = {}


def register(cls: type[Domain]) -> type[Domain]:
    """Class decorator to register a domain plugin."""
    if cls.DOMAIN_ID:
        _registry[cls.DOMAIN_ID] = cls
    return cls


def load_domain(domain_id: str) -> Domain:
    """Load and instantiate a domain plugin by ID."""
    if not _registry:
        _discover()
    if domain_id not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(f"Unknown domain '{domain_id}'. Available: {available}")
    return _registry[domain_id]()


def list_domains() -> list[str]:
    """Return all registered domain IDs."""
    if not _registry:
        _discover()
    return sorted(_registry.keys())


def _discover():
    """Import all domain modules to trigger registration."""
    domains_dir = Path(__file__).parent / "domains"
    if not domains_dir.is_dir():
        return
    for f in domains_dir.glob("*.py"):
        if f.name.startswith("_"):
            continue
        module_name = f"domains.{f.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            print(f"Warning: failed to load domain {module_name}: {e}")
