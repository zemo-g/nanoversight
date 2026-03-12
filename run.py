#!/usr/bin/env python3
"""nanoversight CLI entry point.

Usage:
    python3 run.py --domain sysmon
    python3 run.py --domain sysmon --model llama3.2
    python3 run.py --domain sysmon --llm-url http://localhost:11434/v1/chat/completions
    python3 run.py --domain example --max-cycles 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from domain import list_domains, load_domain  # noqa: E402
from nanoversight import Oversight  # noqa: E402


def make_llm(url: str, model: str, temperature: float, max_tokens: int):
    """Create an LLM callable that talks to an OpenAI-compatible API."""

    def llm(system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"] or ""

    return llm


def main():
    parser = argparse.ArgumentParser(description="nanoversight — autonomous oversight")
    parser.add_argument(
        "--domain", required=True,
        help=f"Domain plugin to use. Available: {', '.join(list_domains())}",
    )
    parser.add_argument(
        "--db", default=None,
        help="Database path (default: data/<domain>.db)",
    )
    parser.add_argument(
        "--llm-url", default="http://localhost:11434/v1/chat/completions",
        help="OpenAI-compatible LLM endpoint URL (default: Ollama)",
    )
    parser.add_argument(
        "--model", default="llama3.2",
        help="Model name to send to the LLM API",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="LLM temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="LLM max tokens",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=0,
        help="Max cycles to run (0 = forever)",
    )
    parser.add_argument(
        "--watch-dir", default=None,
        help="Directory to watch (example domain only)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load domain
    domain = load_domain(args.domain)

    # Pass watch-dir to example domain
    if args.watch_dir and hasattr(domain, "watch_dir"):
        domain.watch_dir = Path(args.watch_dir)
        domain.watch_dir.mkdir(parents=True, exist_ok=True)

    # DB path
    db_path = args.db or f"data/{args.domain}.db"

    # LLM
    llm = make_llm(args.llm_url, args.model, args.temperature, args.max_tokens)

    # Run
    engine = Oversight(domain=domain, llm=llm, db_path=db_path)
    try:
        engine.run(max_cycles=args.max_cycles)
    except KeyboardInterrupt:
        logging.getLogger("nanoversight").info("Interrupted. Goodbye.")


if __name__ == "__main__":
    main()
