"""Docker domain — watches container health and resource usage.

Reads from `docker stats` and `docker ps`. No Docker SDK needed —
pure CLI subprocess calls.

Usage:
    python3 run.py --domain docker
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domain import Domain, register  # noqa: E402


def _run(cmd: list[str]) -> str:
    """Run a command, return stdout. Empty string on failure."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except Exception:
        return ""


@register
class DockerDomain(Domain):
    DOMAIN_ID = "docker"
    DOMAIN_NAME = "Docker Containers"
    SEED_QUESTIONS = [
        "Are any containers using excessive memory?",
        "Is CPU usage distributed evenly across containers?",
        "Are there containers that keep restarting?",
        "Is any container's memory usage trending upward — possible leak?",
        "Are all expected containers running?",
        "Which container is the biggest resource consumer?",
        "Is the overall system impact of containers manageable?",
    ]

    def observe(self) -> dict[str, Any]:
        """Collect container stats via docker CLI."""
        # Check if docker is available
        version = _run(["docker", "version", "--format", "{{.Server.Version}}"])
        if not version:
            return {"status": "docker_unavailable"}

        metrics: dict[str, Any] = {"docker_version": version}

        # Container counts by state
        ps_all = _run(["docker", "ps", "-a", "--format", "{{.State}}"])
        if ps_all:
            states = ps_all.strip().split("\n")
            metrics["containers_total"] = len(states)
            metrics["containers_running"] = states.count("running")
            metrics["containers_exited"] = states.count("exited")
            metrics["containers_restarting"] = states.count("restarting")
        else:
            metrics["containers_total"] = 0
            metrics["containers_running"] = 0

        # Stats for running containers (no-stream = snapshot)
        stats_out = _run([
            "docker", "stats", "--no-stream",
            "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.PIDs}}"
        ])
        if stats_out:
            total_cpu = 0.0
            total_mem_pct = 0.0
            top_cpu_name = ""
            top_cpu_val = 0.0
            top_mem_name = ""
            top_mem_val = 0.0
            container_count = 0

            for line in stats_out.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                name = parts[0]
                try:
                    cpu = float(parts[1].rstrip("%"))
                    mem_pct = float(parts[3].rstrip("%"))
                except ValueError:
                    continue

                container_count += 1
                total_cpu += cpu
                total_mem_pct += mem_pct

                if cpu > top_cpu_val:
                    top_cpu_val = cpu
                    top_cpu_name = name
                if mem_pct > top_mem_val:
                    top_mem_val = mem_pct
                    top_mem_name = name

            metrics["total_cpu_pct"] = round(total_cpu, 1)
            metrics["total_mem_pct"] = round(total_mem_pct, 1)
            if top_cpu_name:
                metrics["top_cpu_container"] = top_cpu_name
                metrics["top_cpu_pct"] = top_cpu_val
            if top_mem_name:
                metrics["top_mem_container"] = top_mem_name
                metrics["top_mem_pct"] = top_mem_val

        # Recently exited containers (last 5)
        exited = _run([
            "docker", "ps", "-a", "--filter", "status=exited",
            "--format", "{{.Names}}\t{{.Status}}", "-n", "5"
        ])
        if exited:
            exit_lines = exited.strip().split("\n")
            metrics["recently_exited"] = len(exit_lines)
            # Check for crash exits (non-zero)
            crash_out = _run([
                "docker", "ps", "-a", "--filter", "status=exited",
                "--filter", "exited=1",
                "--format", "{{.Names}}", "-n", "10"
            ])
            if crash_out:
                metrics["crash_exits"] = len(crash_out.strip().split("\n"))

        return metrics

    def format_context(
        self,
        metrics: dict[str, Any],
        conclusions: list[dict],
        changes: list[str],
    ) -> str:
        ctx = "Docker container monitor\n\n"

        if metrics.get("status") == "docker_unavailable":
            ctx += "Docker is not available or not running.\n"
            return ctx

        ctx += "Current state:\n"
        ctx += f"  Docker version: {metrics.get('docker_version', '?')}\n"
        ctx += f"  Containers: {metrics.get('containers_running', 0)} running"
        ctx += f" / {metrics.get('containers_total', 0)} total\n"
        if metrics.get("containers_restarting", 0) > 0:
            ctx += f"  Restarting: {metrics['containers_restarting']} (possible crash loop)\n"
        if metrics.get("containers_exited", 0) > 0:
            ctx += f"  Exited: {metrics['containers_exited']}\n"

        if "total_cpu_pct" in metrics:
            ctx += f"\n  Total CPU: {metrics['total_cpu_pct']}%\n"
            ctx += f"  Total Memory: {metrics.get('total_mem_pct', '?')}%\n"
        if "top_cpu_container" in metrics:
            ctx += f"  Top CPU: {metrics['top_cpu_container']} ({metrics['top_cpu_pct']}%)\n"
        if "top_mem_container" in metrics:
            ctx += f"  Top MEM: {metrics['top_mem_container']} ({metrics['top_mem_pct']}%)\n"

        if metrics.get("crash_exits", 0) > 0:
            ctx += f"\n  Crash exits (non-zero): {metrics['crash_exits']}\n"

        if changes:
            ctx += "\nChanges since last check:\n"
            for c in changes:
                ctx += f"  - {c}\n"

        if conclusions:
            ctx += "\nSettled knowledge:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"

        return ctx
