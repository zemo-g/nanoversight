"""System monitor domain — your safety net.

Watches CPU, memory, disk, and process health. Thinks about resource
trends, detects anomalies, and warns you before things break.

Works on macOS and Linux out of the box. No dependencies.

Usage:
    python3 run.py --domain sysmon
"""

from __future__ import annotations

import os
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domain import Domain, register  # noqa: E402


def _run(cmd: list[str]) -> str:
    """Run a command, return stdout. Empty string on failure."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return ""


@register
class SysmonDomain(Domain):
    DOMAIN_ID = "sysmon"
    DOMAIN_NAME = "System Monitor"
    SEED_QUESTIONS = [
        "Is memory usage trending up or stable?",
        "Are any processes consuming an unusual amount of CPU?",
        "Is disk space running low on any volume?",
        "Are there signs of a memory leak in any long-running process?",
        "Is the system under more load than usual?",
        "Are there any zombie or stuck processes?",
        "Is swap usage increasing — a sign memory pressure is building?",
        "What is the biggest resource risk right now?",
    ]

    def observe(self) -> dict[str, Any]:
        """Collect system vitals using only stdlib + standard CLI tools."""
        metrics: dict[str, Any] = {}

        # CPU load averages (1m, 5m, 15m)
        try:
            load1, load5, load15 = os.getloadavg()
            metrics["load_1m"] = round(load1, 2)
            metrics["load_5m"] = round(load5, 2)
            metrics["load_15m"] = round(load15, 2)
        except OSError:
            pass

        # CPU count (for context — load > cpu_count means saturated)
        cpu_count = os.cpu_count() or 1
        metrics["cpu_count"] = cpu_count

        # Memory (platform-aware)
        mem = self._memory_info()
        metrics.update(mem)

        # Disk usage on root volume
        try:
            st = os.statvfs("/")
            total_gb = (st.f_blocks * st.f_frsize) / (1024 ** 3)
            free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
            used_pct = round((1 - free_gb / total_gb) * 100, 1) if total_gb > 0 else 0
            metrics["disk_total_gb"] = round(total_gb, 1)
            metrics["disk_free_gb"] = round(free_gb, 1)
            metrics["disk_used_pct"] = used_pct
        except OSError:
            pass

        # Process count
        try:
            pids = [p for p in os.listdir("/proc") if p.isdigit()] if os.path.isdir("/proc") else []
            if pids:
                metrics["process_count"] = len(pids)
            else:
                # macOS fallback
                out = _run(["ps", "-ax", "-o", "pid="])
                if out:
                    metrics["process_count"] = len(out.strip().split("\n"))
        except Exception:
            pass

        # Top CPU consumers
        top_cpu = self._top_processes("cpu")
        if top_cpu:
            metrics["top_cpu_process"] = top_cpu[0][0]
            metrics["top_cpu_pct"] = top_cpu[0][1]

        # Top memory consumers
        top_mem = self._top_processes("mem")
        if top_mem:
            metrics["top_mem_process"] = top_mem[0][0]
            metrics["top_mem_pct"] = top_mem[0][1]

        # Uptime in hours
        uptime_s = self._uptime_seconds()
        if uptime_s is not None:
            metrics["uptime_hours"] = round(uptime_s / 3600, 1)

        # Zombie processes
        zombies = _run(["ps", "-ax", "-o", "state="])
        if zombies:
            metrics["zombie_count"] = zombies.count("Z")

        return metrics

    def format_context(
        self,
        metrics: dict[str, Any],
        conclusions: list[dict],
        changes: list[str],
    ) -> str:
        cpu_count = metrics.get("cpu_count", "?")
        ctx = f"System monitor — {cpu_count} CPUs\n\n"

        ctx += "Current vitals:\n"
        if "load_1m" in metrics:
            ctx += f"  Load: {metrics['load_1m']} (1m) / {metrics['load_5m']} (5m) / {metrics['load_15m']} (15m)\n"
            load_ratio = metrics["load_1m"] / max(metrics.get("cpu_count", 1), 1)
            if load_ratio > 1.0:
                ctx += f"  ⚠ CPU saturated — load/core ratio: {load_ratio:.1f}\n"
        if "mem_used_pct" in metrics:
            ctx += f"  Memory: {metrics['mem_used_pct']}% used"
            if "mem_free_mb" in metrics:
                ctx += f" ({metrics['mem_free_mb']} MB free)"
            ctx += "\n"
        if "swap_used_mb" in metrics:
            ctx += f"  Swap: {metrics['swap_used_mb']} MB used\n"
        if "disk_used_pct" in metrics:
            ctx += f"  Disk: {metrics['disk_used_pct']}% used ({metrics.get('disk_free_gb', '?')} GB free)\n"
        if "process_count" in metrics:
            ctx += f"  Processes: {metrics['process_count']}"
            if metrics.get("zombie_count", 0) > 0:
                ctx += f" ({metrics['zombie_count']} zombies)"
            ctx += "\n"
        if "uptime_hours" in metrics:
            ctx += f"  Uptime: {metrics['uptime_hours']}h\n"
        if "top_cpu_process" in metrics:
            ctx += f"  Top CPU: {metrics['top_cpu_process']} ({metrics['top_cpu_pct']}%)\n"
        if "top_mem_process" in metrics:
            ctx += f"  Top MEM: {metrics['top_mem_process']} ({metrics['top_mem_pct']}%)\n"

        if changes:
            ctx += "\nChanges since last check:\n"
            for c in changes:
                ctx += f"  - {c}\n"

        if conclusions:
            ctx += "\nSettled knowledge:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"

        return ctx

    # ── Internals ────────────────────────────────────────────────────────

    def _memory_info(self) -> dict[str, Any]:
        """Get memory stats. Works on macOS and Linux."""
        # Try Linux /proc/meminfo first
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
                total = info.get("MemTotal", 0)
                available = info.get("MemAvailable", info.get("MemFree", 0))
                swap_total = info.get("SwapTotal", 0)
                swap_free = info.get("SwapFree", 0)
                return {
                    "mem_total_mb": total // 1024,
                    "mem_free_mb": available // 1024,
                    "mem_used_pct": round((1 - available / max(total, 1)) * 100, 1),
                    "swap_used_mb": (swap_total - swap_free) // 1024,
                }
        except FileNotFoundError:
            pass

        # macOS fallback — vm_stat + sysctl
        out = _run(["vm_stat"])
        if out:
            pages: dict[str, int] = {}
            for line in out.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip().rstrip(".")
                    try:
                        pages[key.strip()] = int(val)
                    except ValueError:
                        pass
            page_size = 16384  # Apple Silicon default
            free = pages.get("Pages free", 0)
            inactive = pages.get("Pages inactive", 0)
            speculative = pages.get("Pages speculative", 0)
            available_pages = free + inactive + speculative

            # Total from sysctl
            total_bytes = _run(["sysctl", "-n", "hw.memsize"])
            total_mb = int(total_bytes) // (1024 * 1024) if total_bytes else 0
            free_mb = (available_pages * page_size) // (1024 * 1024)

            result: dict[str, Any] = {
                "mem_total_mb": total_mb,
                "mem_free_mb": free_mb,
                "mem_used_pct": round((1 - free_mb / max(total_mb, 1)) * 100, 1),
            }

            # Swap — parse "total = X  used = Y  free = Z"
            swap_out = _run(["sysctl", "-n", "vm.swapusage"])
            if swap_out and "used" in swap_out:
                for segment in swap_out.split("  "):
                    segment = segment.strip()
                    if segment.startswith("used"):
                        val = segment.split("=")[1].strip().rstrip("M").strip()
                        try:
                            result["swap_used_mb"] = round(float(val))
                        except ValueError:
                            pass

            return result

        return {}

    def _top_processes(self, sort_by: str) -> list[tuple[str, float]]:
        """Get top 3 processes by CPU or memory."""
        flag = "-%cpu" if sort_by == "cpu" else "-%mem"
        out = _run(["ps", "-axo", f"comm=,{flag[1:]}=", "-r"])
        if not out:
            return []
        results = []
        for line in out.strip().split("\n")[:3]:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                name = parts[0].strip().split("/")[-1]  # basename
                try:
                    pct = float(parts[1])
                    results.append((name, pct))
                except ValueError:
                    pass
        return results

    def _uptime_seconds(self) -> float | None:
        """Get system uptime in seconds."""
        # Linux
        try:
            with open("/proc/uptime") as f:
                return float(f.read().split()[0])
        except FileNotFoundError:
            pass
        # macOS — format: { sec = 1234567890, usec = 0 }
        out = _run(["sysctl", "-n", "kern.boottime"])
        if out and "sec" in out:
            try:
                sec_str = out.split("sec =")[1].split(",")[0].strip()
                boot_time = int(sec_str)
                return _time.time() - boot_time
            except (IndexError, ValueError):
                pass
        return None
