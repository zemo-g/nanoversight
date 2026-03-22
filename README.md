# nanoversight

[![tests](https://github.com/zemo-g/nanoversight/actions/workflows/test.yml/badge.svg)](https://github.com/zemo-g/nanoversight/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#)

Autonomous oversight for any system. One file, one loop, one LLM.

Point it at anything — your server, your application, your infrastructure — and it starts thinking. It observes metrics, asks questions, forms conclusions, and grows new questions it didn't start with. Like having a junior engineer watching your dashboards 24/7, except it never sleeps and it gets smarter over time.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest thing that works.

## Quickstart

```bash
# Install Ollama (or any OpenAI-compatible LLM server)
# https://ollama.ai

# Run system monitor — watches CPU, memory, disk, processes
python3 run.py --domain sysmon

# Use a specific model
python3 run.py --domain sysmon --model llama3.2

# Point at a different LLM endpoint
python3 run.py --domain sysmon --llm-url http://localhost:8080/v1/chat/completions --model my-model

# Run for 10 cycles then stop
python3 run.py --domain sysmon --max-cycles 10 -v
```

No dependencies. Pure Python stdlib.

## How it works

Every 60 seconds:

```
observe()    → collect metrics from the domain
think()      → pick a question, ask the LLM, store the thought
reflect()    → statistical trend detection + LLM pattern extraction → conclusions
grow()       → generate new questions, retire stale ones
prune()      → delete old data
```

The engine starts with seed questions from the domain, then evolves its own. Questions that never produce insights get retired. Conclusions that appear repeatedly at high confidence become "settled knowledge" — injected into every future LLM call as permanent context.

### What makes it work

- **Confidence calibration** — the LLM is prompted to be harsh (most honest scores are 0.3-0.5). Prevents the system from convincing itself of noise.
- **Repetition penalty** — if a thought is too similar to the previous one, confidence is capped. Forces the system to push deeper or admit it's stuck.
- **Settle logic** — same pattern appearing 2+ times at 0.85+ confidence becomes settled knowledge. The system builds on what it knows.
- **Question lifecycle** — over-asked questions with no progress get retired. The system stops wasting cycles on dead ends.

## Architecture

```
nanoversight/
  nanoversight.py    Core engine — one class, one loop
  domain.py          Domain plugin ABC + registry
  config.py          Raw constants
  run.py             CLI entry point
  domains/
    sysmon.py        System monitor (CPU, memory, disk, processes)
    github.py        GitHub repo watcher (stars, forks, issues, traffic)
    docker.py        Docker container health and resource usage
    webserver.py     HTTP endpoint latency and availability
    example.py       Minimal example (directory watcher)
```

### DB schema (5 tables)

| Table | Purpose |
|-------|---------|
| observations | Timestamped metrics from domain.observe() |
| thoughts | Question + LLM answer + confidence per cycle |
| conclusions | Extracted patterns with confidence, evidence, settled flag |
| questions | Active question lifecycle (times_asked, retired) |
| actions | Recommendations attempted + results |

## Built-in domains

| Domain | What it watches | Command |
|--------|----------------|---------|
| `sysmon` | CPU, memory, disk, swap, processes | `python3 run.py --domain sysmon` |
| `github` | Stars, forks, issues, PRs, traffic | `NANOVERSIGHT_GITHUB_REPO=owner/repo python3 run.py --domain github` |
| `docker` | Container health, CPU/memory per container | `python3 run.py --domain docker` |
| `webserver` | HTTP status, latency, availability | `NANOVERSIGHT_URLS=https://example.com python3 run.py --domain webserver` |
| `example` | Directory file changes (template) | `python3 run.py --domain example` |

## Write your own domain

A domain plugin is ~80 lines. Two required methods:

```python
from domain import Domain, register

@register
class MyDomain(Domain):
    DOMAIN_ID = "mydomain"
    DOMAIN_NAME = "My Application"
    SEED_QUESTIONS = [
        "Is the system healthy?",
        "Are there any concerning trends?",
    ]

    def observe(self) -> dict:
        """Collect metrics. Return a flat dict of name → value."""
        return {
            "active_users": get_active_users(),
            "error_rate": get_error_rate(),
            "p99_latency_ms": get_p99(),
        }

    def format_context(self, metrics, conclusions, changes) -> str:
        """Build a string the LLM will read before answering."""
        ctx = "My Application Status\n"
        for k, v in metrics.items():
            ctx += f"  {k}: {v}\n"
        if conclusions:
            ctx += "\nKnown patterns:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"
        return ctx
```

Drop it in `domains/`, and `python3 run.py --domain mydomain` works.

Two optional methods for domains that can act:

```python
    def parse_recommendation(self, llm_output: str) -> dict | None:
        """Extract a structured recommendation from LLM output."""
        ...

    def act(self, recommendation: dict) -> bool:
        """Execute it. Return True if acted on."""
        ...
```

## LLM interface

The engine takes a single callable:

```python
llm: Callable[[str, str], str]  # (system_prompt, user_prompt) → response
```

The built-in `run.py` wraps any OpenAI-compatible API (Ollama, llama.cpp, vLLM, LM Studio, etc). For custom backends:

```python
from nanoversight import Oversight
from domain import load_domain

def my_llm(system: str, user: str) -> str:
    # Call your LLM however you want
    return response

engine = Oversight(
    domain=load_domain("sysmon"),
    llm=my_llm,
    db_path="data/sysmon.db",
)
engine.run()
```

## Configuration

All constants live in `config.py`. The defaults work. Tune if needed:

| Constant | Default | What |
|----------|---------|------|
| CYCLE_INTERVAL | 60 | Seconds between cycles |
| REFLECT_EVERY | 5 | Cycles between reflect passes |
| GROW_EVERY | 15 | Cycles between question growth |
| MAX_ACTIVE_QUESTIONS | 100 | Hard cap on active questions |
| SETTLED_MIN_CONFIDENCE | 0.85 | Minimum confidence to settle a conclusion |
| REPETITION_OVERLAP_THRESHOLD | 0.6 | Word overlap that triggers repetition penalty |

## License

MIT
