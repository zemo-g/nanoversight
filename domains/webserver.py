"""Web server domain — watches HTTP endpoints for health and latency.

Pings a list of URLs and tracks response time, status codes, and
availability. No log parsing needed — works with any web server.

Usage:
    python3 run.py --domain webserver
    NANOVERSIGHT_URLS=https://example.com,https://api.example.com/health python3 run.py --domain webserver
"""

from __future__ import annotations

import os
import sys
import time as _time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domain import Domain, register  # noqa: E402

DEFAULT_URLS = "http://localhost:8080/health"


def _probe(url: str, timeout: int = 10) -> dict[str, Any]:
    """Probe a URL. Returns status, latency, and any error."""
    t0 = _time.monotonic()
    try:
        req = Request(url, headers={"User-Agent": "nanoversight"})
        with urlopen(req, timeout=timeout) as resp:
            latency_ms = round((_time.monotonic() - t0) * 1000)
            size = len(resp.read())
            return {
                "status": resp.status,
                "latency_ms": latency_ms,
                "size_bytes": size,
                "error": None,
            }
    except Exception as e:
        latency_ms = round((_time.monotonic() - t0) * 1000)
        return {
            "status": 0,
            "latency_ms": latency_ms,
            "size_bytes": 0,
            "error": str(e)[:100],
        }


@register
class WebserverDomain(Domain):
    DOMAIN_ID = "webserver"
    DOMAIN_NAME = "Web Server Monitor"
    SEED_QUESTIONS = [
        "Are all endpoints responding within acceptable latency?",
        "Is any endpoint's response time trending upward?",
        "Are there intermittent failures on any endpoint?",
        "Is response size changing — could indicate errors or truncation?",
        "Which endpoint is the slowest and why might that be?",
        "Is there a pattern to when failures occur?",
        "Are all endpoints returning 200 or are some degraded?",
    ]

    def __init__(self):
        url_str = os.environ.get("NANOVERSIGHT_URLS", DEFAULT_URLS)
        self.urls = [u.strip() for u in url_str.split(",") if u.strip()]

    def observe(self) -> dict[str, Any]:
        """Probe all configured URLs."""
        metrics: dict[str, Any] = {
            "endpoints": len(self.urls),
        }

        total_latency = 0
        healthy = 0
        unhealthy = 0
        slowest_url = ""
        slowest_ms = 0

        for i, url in enumerate(self.urls):
            result = _probe(url)
            # Use short key based on index to keep metrics flat
            tag = url.split("//")[-1].split("/")[0].replace(".", "_").replace(":", "_")
            if i > 0:
                tag = f"{tag}_{i}"

            metrics[f"{tag}_status"] = result["status"]
            metrics[f"{tag}_latency_ms"] = result["latency_ms"]
            metrics[f"{tag}_size"] = result["size_bytes"]

            if result["error"]:
                metrics[f"{tag}_error"] = result["error"]
                unhealthy += 1
            elif result["status"] >= 400:
                unhealthy += 1
            else:
                healthy += 1

            total_latency += result["latency_ms"]
            if result["latency_ms"] > slowest_ms:
                slowest_ms = result["latency_ms"]
                slowest_url = url

        metrics["healthy"] = healthy
        metrics["unhealthy"] = unhealthy
        metrics["avg_latency_ms"] = round(total_latency / max(len(self.urls), 1))
        if slowest_url:
            metrics["slowest_endpoint"] = slowest_url.split("//")[-1][:40]
            metrics["slowest_ms"] = slowest_ms

        return metrics

    def format_context(
        self,
        metrics: dict[str, Any],
        conclusions: list[dict],
        changes: list[str],
    ) -> str:
        ctx = f"Web server monitor — {metrics.get('endpoints', 0)} endpoints\n\n"

        ctx += "Health:\n"
        ctx += f"  Healthy: {metrics.get('healthy', 0)} / {metrics.get('endpoints', 0)}\n"
        if metrics.get("unhealthy", 0) > 0:
            ctx += f"  Unhealthy: {metrics['unhealthy']}\n"
        ctx += f"  Avg latency: {metrics.get('avg_latency_ms', '?')} ms\n"
        if "slowest_endpoint" in metrics:
            ctx += f"  Slowest: {metrics['slowest_endpoint']} ({metrics['slowest_ms']} ms)\n"

        # Per-endpoint details
        ctx += "\nEndpoints:\n"
        for url in self.urls:
            tag = url.split("//")[-1].split("/")[0].replace(".", "_").replace(":", "_")
            status = metrics.get(f"{tag}_status", "?")
            latency = metrics.get(f"{tag}_latency_ms", "?")
            error = metrics.get(f"{tag}_error")
            line = f"  {url} → {status} ({latency} ms)"
            if error:
                line += f" ERROR: {error}"
            ctx += line + "\n"

        if changes:
            ctx += "\nChanges since last check:\n"
            for c in changes:
                ctx += f"  - {c}\n"

        if conclusions:
            ctx += "\nSettled knowledge:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"

        return ctx
