"""Example domain — watches a directory for file changes.

Demonstrates the minimal domain plugin contract. Good for testing
nanoversight without any external dependencies.

Usage:
    python3 run.py --domain example
    python3 run.py --domain example --watch-dir /tmp/nanoversight-test
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domain import Domain, register  # noqa: E402


@register
class ExampleDomain(Domain):
    DOMAIN_ID = "example"
    DOMAIN_NAME = "Directory Watcher"
    SEED_QUESTIONS = [
        "What types of files are most common in this directory?",
        "Is the directory growing or stable?",
        "Are there any unusually large files?",
        "How frequently are files being modified?",
        "Are there any patterns in file naming or organization?",
    ]

    def __init__(self, watch_dir: str = "/tmp/nanoversight-test"):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

    def observe(self) -> dict[str, Any]:
        """Count files, total size, newest mod time."""
        files = list(self.watch_dir.iterdir())
        total_size = 0
        newest_mtime = 0.0
        extensions: dict[str, int] = {}
        for f in files:
            if f.is_file():
                stat = f.stat()
                total_size += stat.st_size
                newest_mtime = max(newest_mtime, stat.st_mtime)
                ext = f.suffix or "(none)"
                extensions[ext] = extensions.get(ext, 0) + 1

        top_ext = "(empty)"
        if extensions:
            top_ext = max(extensions, key=lambda k: extensions[k])

        return {
            "file_count": len([f for f in files if f.is_file()]),
            "dir_count": len([f for f in files if f.is_dir()]),
            "total_size_kb": round(total_size / 1024, 1),
            "newest_age_sec": round(_time.time() - newest_mtime) if newest_mtime else -1,
            "top_extension": top_ext,
        }

    def format_context(
        self,
        metrics: dict[str, Any],
        conclusions: list[dict],
        changes: list[str],
    ) -> str:
        ctx = f"Watching directory: {self.watch_dir}\n\n"
        ctx += "Current state:\n"
        for k, v in metrics.items():
            ctx += f"  {k}: {v}\n"

        if changes:
            ctx += "\nChanges since last observation:\n"
            for c in changes:
                ctx += f"  - {c}\n"

        if conclusions:
            ctx += "\nSettled knowledge:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"

        return ctx
