"""GitHub repo watcher domain — watches a repo's pulse.

Tracks stars, forks, issues, PRs, and recent activity using the
GitHub API (no auth required for public repos, rate-limited to 60/hr).

Usage:
    python3 run.py --domain github
    NANOVERSIGHT_GITHUB_REPO=owner/repo python3 run.py --domain github
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domain import Domain, register  # noqa: E402

DEFAULT_REPO = "zemo-g/nanoversight"
API_BASE = "https://api.github.com"


def _gh_get(path: str) -> dict | list | None:
    """GET from GitHub API. Returns parsed JSON or None on failure."""
    url = f"{API_BASE}{path}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "nanoversight",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


@register
class GitHubDomain(Domain):
    DOMAIN_ID = "github"
    DOMAIN_NAME = "GitHub Repository"
    SEED_QUESTIONS = [
        "Is the repo gaining traction or stagnant?",
        "Are issues being opened faster than they're being closed?",
        "What does the star growth rate look like?",
        "Are there any pull requests that need attention?",
        "Is the community engaging or just starring?",
        "What content or changes drive the most stars?",
        "Is the fork-to-star ratio healthy?",
    ]

    def __init__(self):
        self.repo = os.environ.get("NANOVERSIGHT_GITHUB_REPO", DEFAULT_REPO)

    def observe(self) -> dict[str, Any]:
        """Fetch repo stats from GitHub API."""
        repo_data = _gh_get(f"/repos/{self.repo}")
        if not repo_data or isinstance(repo_data, list):
            return {"status": "api_error", "repo": self.repo}

        metrics: dict[str, Any] = {
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "open_issues": repo_data.get("open_issues_count", 0),
            "watchers": repo_data.get("subscribers_count", 0),
            "size_kb": repo_data.get("size", 0),
        }

        # Open PRs
        prs = _gh_get(f"/repos/{self.repo}/pulls?state=open&per_page=5")
        if isinstance(prs, list):
            metrics["open_prs"] = len(prs)

        # Recent issues (last 5)
        issues = _gh_get(f"/repos/{self.repo}/issues?state=all&per_page=5&sort=created&direction=desc")
        if isinstance(issues, list):
            # GitHub API returns PRs in issues endpoint too — filter them
            real_issues = [i for i in issues if "pull_request" not in i]
            metrics["recent_issues"] = len(real_issues)

        # Recent commits (last 5)
        commits = _gh_get(f"/repos/{self.repo}/commits?per_page=5")
        if isinstance(commits, list):
            metrics["recent_commits"] = len(commits)
            if commits:
                metrics["last_commit_by"] = (
                    commits[0].get("commit", {}).get("author", {}).get("name", "unknown")
                )

        # Traffic (requires auth — will be 0 without GITHUB_TOKEN)
        traffic = _gh_get(f"/repos/{self.repo}/traffic/views")
        if isinstance(traffic, dict):
            metrics["views_14d"] = traffic.get("count", 0)
            metrics["unique_visitors_14d"] = traffic.get("uniques", 0)

        clones = _gh_get(f"/repos/{self.repo}/traffic/clones")
        if isinstance(clones, dict):
            metrics["clones_14d"] = clones.get("count", 0)

        return metrics

    def format_context(
        self,
        metrics: dict[str, Any],
        conclusions: list[dict],
        changes: list[str],
    ) -> str:
        repo = self.repo
        ctx = f"GitHub repository: {repo}\n\n"

        if metrics.get("status") == "api_error":
            ctx += "API request failed — rate limited or repo unavailable.\n"
            return ctx

        ctx += "Current stats:\n"
        ctx += f"  Stars: {metrics.get('stars', '?')}\n"
        ctx += f"  Forks: {metrics.get('forks', '?')}\n"
        ctx += f"  Watchers: {metrics.get('watchers', '?')}\n"
        ctx += f"  Open issues: {metrics.get('open_issues', '?')}\n"
        if "open_prs" in metrics:
            ctx += f"  Open PRs: {metrics['open_prs']}\n"
        if "views_14d" in metrics:
            ctx += f"  Views (14d): {metrics['views_14d']} ({metrics.get('unique_visitors_14d', '?')} unique)\n"
        if "clones_14d" in metrics:
            ctx += f"  Clones (14d): {metrics['clones_14d']}\n"
        if "recent_commits" in metrics:
            ctx += f"  Recent commits: {metrics['recent_commits']}\n"
        if "last_commit_by" in metrics:
            ctx += f"  Last commit by: {metrics['last_commit_by']}\n"

        stars = metrics.get("stars", 0)
        forks = metrics.get("forks", 0)
        if stars > 0:
            ratio = round(forks / stars * 100, 1)
            ctx += f"  Fork/star ratio: {ratio}%\n"

        if changes:
            ctx += "\nChanges since last check:\n"
            for c in changes:
                ctx += f"  - {c}\n"

        if conclusions:
            ctx += "\nSettled knowledge:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"

        return ctx
