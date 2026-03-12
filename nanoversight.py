"""nanoversight — autonomous oversight in one file.

Single-threaded. No async. No frameworks. Pure stdlib.

Core loop (every cycle):
    observe   — collect metrics from the domain
    think     — pick a question, ask the LLM, store the thought
    reflect   — statistical + LLM pattern extraction into conclusions
    grow      — generate new questions, retire stale ones
    prune     — delete old observations and thoughts
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from config import (
    CYCLE_INTERVAL,
    DEDUP_OVERLAP_THRESHOLD,
    DEDUP_WINDOW_MINUTES,
    GROW_EVERY,
    MAX_ACTIVE_QUESTIONS,
    MAX_SETTLED,
    OBSERVATIONS_RETAIN_HOURS,
    QUESTION_RETIRE_THRESHOLD,
    REFLECT_EVERY,
    REPETITION_CONFIDENCE_CAP,
    REPETITION_OVERLAP_THRESHOLD,
    SETTLED_MIN_CONFIDENCE,
    SETTLED_THRESHOLD,
    STAGNATION_MIN_ASKS,
    THOUGHTS_RETAIN_DAYS,
)
from domain import Domain

log = logging.getLogger("nanoversight")


# ── Text utilities ───────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "is", "at", "on", "a", "an", "and", "or", "for", "in",
    "to", "of", "it", "be", "this", "that", "with", "are", "was",
    "has", "have", "not", "but", "if", "as", "by", "from",
}


def word_overlap(a: str, b: str) -> float:
    """Word-level Jaccard overlap between two strings."""
    def words(s: str):
        return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) - _STOP_WORDS
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def extract_confidence(raw: str) -> tuple[str, float]:
    """Parse CONFIDENCE: marker from LLM output. Returns (answer, confidence)."""
    lines = raw.strip().split("\n")
    answer_lines = []
    confidence = 0.4
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CONFIDENCE:"):
            try:
                val = float(stripped.split(":", 1)[1].strip())
                confidence = max(0.1, min(1.0, val))
            except (ValueError, IndexError):
                pass
        else:
            answer_lines.append(line)
    return "\n".join(answer_lines).strip(), confidence


def parse_conclusions(raw: str) -> list[tuple[str, str, float, str]]:
    """Parse PATTERN/CONCLUSION/EVIDENCE/CONFIDENCE blocks from LLM output.

    Returns list of (pattern, conclusion, confidence, evidence).
    """
    if not raw or raw.strip().upper() == "NONE":
        return []
    results = []
    for block in raw.split("---"):
        block = block.strip()
        if not block:
            continue
        pattern = conclusion = evidence = None
        confidence = 0.5
        for line in block.split("\n"):
            line = line.strip()
            upper = line.upper()
            if upper.startswith("PATTERN:"):
                pattern = line.split(":", 1)[1].strip().lower().replace(" ", "_")[:60]
            elif upper.startswith("CONCLUSION:"):
                conclusion = line.split(":", 1)[1].strip()
            elif upper.startswith("EVIDENCE:"):
                evidence = line.split(":", 1)[1].strip()
            elif upper.startswith("CONFIDENCE:"):
                try:
                    confidence = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except ValueError:
                    confidence = 0.5
        if pattern and conclusion:
            results.append((pattern, conclusion, confidence, evidence or ""))
    return results[:4]


# ── DB setup ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thoughts (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'llm'
);

CREATE TABLE IF NOT EXISTS conclusions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    pattern TEXT NOT NULL,
    conclusion TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence TEXT DEFAULT '',
    evidence_count INTEGER DEFAULT 1,
    settled INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL UNIQUE,
    origin TEXT DEFAULT 'seed',
    times_asked INTEGER DEFAULT 0,
    best_confidence REAL DEFAULT 0.0,
    retired INTEGER DEFAULT 0,
    created TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    recommendation_json TEXT NOT NULL,
    acted INTEGER DEFAULT 0,
    result TEXT DEFAULT ''
);
"""


def _open_db(path: Path) -> sqlite3.Connection:
    """Open (or create) the nanoversight database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.executescript(_SCHEMA)
    return db


# ── Core engine ─────────────────────────────────────────────────────────────

class Oversight:
    """Autonomous oversight engine. One class, one loop."""

    def __init__(
        self,
        domain: Domain,
        llm: Callable[[str, str], str],
        db_path: Path | str = "data/nanoversight.db",
    ):
        self.domain = domain
        self.llm = llm  # (system_prompt, user_prompt) → response
        self.db = _open_db(Path(db_path))
        self.cycle = 0
        self.prev_metrics: dict[str, Any] = {}
        self._seed_questions()

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ── Seed questions ──────────────────────────────────────────────────

    def _seed_questions(self):
        """Insert domain seed questions if the questions table is empty."""
        count = self.db.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        if count > 0:
            return
        now = self._now()
        for q in self.domain.SEED_QUESTIONS:
            self.db.execute(
                "INSERT OR IGNORE INTO questions (question, origin, created) VALUES (?, 'seed', ?)",
                (q, now),
            )
        self.db.commit()

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self, max_cycles: int = 0):
        """Run the oversight loop. max_cycles=0 means run forever."""
        log.info("nanoversight starting — domain=%s", self.domain.DOMAIN_ID)
        while True:
            self.cycle += 1
            t0 = time.monotonic()
            try:
                self._cycle()
            except Exception as e:
                log.error("Cycle %d failed: %s", self.cycle, e)
            elapsed = time.monotonic() - t0
            log.info(
                "Cycle %d complete in %.1fs", self.cycle, elapsed,
            )
            if max_cycles and self.cycle >= max_cycles:
                log.info("Reached max_cycles=%d, stopping.", max_cycles)
                break
            sleep_time = max(0, CYCLE_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _cycle(self):
        """One full oversight cycle."""
        # Observe
        metrics = self.domain.observe()
        now = self._now()
        self.db.execute(
            "INSERT INTO observations (timestamp, metrics_json) VALUES (?, ?)",
            (now, json.dumps(metrics)),
        )
        self.db.commit()

        # Detect changes
        changes = self._detect_changes(metrics)
        self.prev_metrics = metrics

        # Think
        self._think(changes)

        # Reflect
        if self.cycle % REFLECT_EVERY == 0:
            self._reflect()

        # Grow
        if self.cycle % GROW_EVERY == 0:
            self._grow()

        # Prune
        self._prune()

    # ── Change detection ────────────────────────────────────────────────

    def _detect_changes(self, metrics: dict) -> list[str]:
        """Compare current metrics to previous, return human-readable changes."""
        if not self.prev_metrics:
            return ["First observation — no prior data."]
        changes = []
        for key, val in metrics.items():
            prev = self.prev_metrics.get(key)
            if prev is None:
                changes.append(f"New metric: {key} = {val}")
            elif isinstance(val, (int, float)) and isinstance(prev, (int, float)):
                if val != prev:
                    delta = val - prev
                    direction = "up" if delta > 0 else "down"
                    changes.append(f"{key}: {prev} → {val} ({direction} {abs(delta):.4g})")
            elif val != prev:
                changes.append(f"{key} changed: {prev} → {val}")
        return changes

    # ── Think ───────────────────────────────────────────────────────────

    def _think(self, changes: list[str]):
        """Pick a question, ask the LLM, store the thought."""
        row = self._pick_question()
        if not row:
            return
        qid, question = row

        # Build context
        settled = self._settled_conclusions()
        context = self.domain.format_context(
            self.prev_metrics, settled, changes,
        )

        # Recall past answers to this question
        past = self.db.execute(
            "SELECT answer, confidence, cycle FROM thoughts "
            "WHERE question = ? ORDER BY id DESC LIMIT 5",
            (question,),
        ).fetchall()
        if past:
            context += "\n\nMy previous thoughts on this question:\n"
            for ans, conf, cyc in reversed(past):
                context += f'  [cycle {cyc}, {conf:.1f}] "{ans[:200]}"\n'
            context += "Build on your previous thinking. Don't repeat yourself. Push deeper.\n"

        # Quality rules + confidence calibration
        context += "\n\nAnswer this question. Think for yourself. Be direct."
        context += "\nQUALITY RULES:"
        context += "\n- Say 'I don't have enough data yet' if that's true. Silence > noise."
        context += "\n- Repetition is not evidence. If your answer restates previous thoughts, cap confidence at 0.3."
        context += "\n- One sharp insight beats three vague observations."

        context += "\n\nAfter your answer, on a new line write CONFIDENCE: followed by a number 0.0-1.0"
        context += "\nCalibration (be HARSH — most thoughts are 0.3-0.5):"
        context += "\n  0.0-0.2: Speculation. No supporting data."
        context += "\n  0.3-0.4: I see something but can't prove it. Most common honest score."
        context += "\n  0.5-0.6: I can point to 3+ specific data points that support this."
        context += "\n  0.7-0.8: Clear pattern across 10+ observations. I can explain the mechanism."
        context += "\n  0.85+: Would stake money on this. Cite exact evidence or don't go here."
        context += "\nIf you're above 0.5, you MUST cite specific evidence. No evidence = 0.4 max."

        context += f"\n\n{question}"

        # LLM call
        system = (
            f"You are an autonomous oversight system watching '{self.domain.DOMAIN_NAME}'. "
            "You think for yourself. You are direct, concise, and honest. "
            "You never pad answers. You speak in 1-3 sentences unless the thought demands more."
        )
        try:
            raw = self.llm(system, context)
        except Exception as e:
            log.warning("LLM call failed: %s", e)
            return
        if not raw or not raw.strip():
            return

        answer, confidence = extract_confidence(raw)

        # Repetition penalty
        if past:
            last_answer = past[0][0]
            overlap = word_overlap(answer, last_answer)
            if overlap > REPETITION_OVERLAP_THRESHOLD:
                confidence = min(confidence, REPETITION_CONFIDENCE_CAP)
                log.debug("Repetition penalty: overlap=%.2f, capped to %.2f", overlap, confidence)

        # Store thought
        now = self._now()
        self.db.execute(
            "INSERT INTO thoughts (timestamp, cycle, question, answer, confidence, source) "
            "VALUES (?, ?, ?, ?, ?, 'llm')",
            (now, self.cycle, question, answer, confidence),
        )

        # Update question stats
        self.db.execute(
            "UPDATE questions SET times_asked = times_asked + 1, "
            "best_confidence = MAX(best_confidence, ?) WHERE id = ?",
            (confidence, qid),
        )
        self.db.commit()

        # Check for recommendation
        rec = self.domain.parse_recommendation(answer)
        if rec:
            acted = self.domain.act(rec)
            self.db.execute(
                "INSERT INTO actions (timestamp, cycle, recommendation_json, acted, result) "
                "VALUES (?, ?, ?, ?, ?)",
                (now, self.cycle, json.dumps(rec), int(acted), "acted" if acted else "advisory"),
            )
            self.db.commit()

        log.info(
            "Think [%d]: Q=\"%.50s\" → conf=%.2f",
            self.cycle, question, confidence,
        )

    def _pick_question(self) -> tuple[int, str] | None:
        """Pick the least-recently-asked non-retired question."""
        row = self.db.execute(
            "SELECT id, question FROM questions WHERE retired = 0 "
            "ORDER BY times_asked ASC, id ASC LIMIT 1"
        ).fetchone()
        return row if row else None

    def _settled_conclusions(self) -> list[dict]:
        """Return settled conclusions for context injection."""
        rows = self.db.execute(
            "SELECT pattern, conclusion, confidence FROM conclusions "
            "WHERE settled = 1 ORDER BY confidence DESC LIMIT ?",
            (MAX_SETTLED,),
        ).fetchall()
        return [
            {"pattern": r[0], "conclusion": r[1], "confidence": r[2]}
            for r in rows
        ]

    # ── Reflect ─────────────────────────────────────────────────────────

    def _reflect(self):
        """Extract conclusions from recent observations using statistical
        detectors + LLM pattern extraction."""
        recent = self.db.execute(
            "SELECT metrics_json FROM observations ORDER BY id DESC LIMIT 10"
        ).fetchall()
        if len(recent) < 3:
            return

        metrics_history = []
        for (mj,) in recent:
            try:
                metrics_history.append(json.loads(mj))
            except json.JSONDecodeError:
                continue

        new_conclusions: list[tuple[str, str, float, str]] = []

        # Statistical detectors — generic, work on any numeric metric
        if len(metrics_history) >= 3:
            # Collect all numeric keys
            numeric_keys = set()
            for m in metrics_history:
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        numeric_keys.add(k)

            for key in numeric_keys:
                values = [m.get(key) for m in metrics_history if key in m]
                values = [v for v in values if isinstance(v, (int, float))]
                if len(values) < 3:
                    continue

                # Monotonic rise
                if all(values[i] >= values[i + 1] for i in range(min(3, len(values) - 1))):
                    if values[0] > values[-1]:
                        new_conclusions.append((
                            f"{key}_rising",
                            f"{key} has been rising: {values[-1]} → {values[0]}",
                            min(0.7, abs(values[0] - values[-1]) / max(abs(values[-1]), 1) * 0.5 + 0.3),
                            f"{len(values)} readings",
                        ))

                # Monotonic decline
                if all(values[i] <= values[i + 1] for i in range(min(3, len(values) - 1))):
                    if values[0] < values[-1]:
                        new_conclusions.append((
                            f"{key}_declining",
                            f"{key} has been declining: {values[-1]} → {values[0]}",
                            min(0.7, abs(values[0] - values[-1]) / max(abs(values[-1]), 1) * 0.5 + 0.3),
                            f"{len(values)} readings",
                        ))

                # Flatlined (no change across all readings)
                if len(set(values)) == 1 and len(values) >= 5:
                    new_conclusions.append((
                        f"{key}_flatlined",
                        f"{key} has been flat at {values[0]} across {len(values)} readings.",
                        0.5,
                        f"{len(values)} identical readings",
                    ))

        # LLM pattern extraction
        settled = self._settled_conclusions()
        context = self.domain.format_context(
            metrics_history[0] if metrics_history else {},
            settled,
            [f"Reflecting on {len(metrics_history)} recent observations"],
        )

        # Add recent thoughts for the LLM to synthesize
        recent_thoughts = self.db.execute(
            "SELECT question, answer, confidence FROM thoughts "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
        if recent_thoughts:
            context += "\n\nRecent thoughts:\n"
            for q, a, c in recent_thoughts:
                context += f"  [{c:.1f}] Q: {q}\n      A: {a[:200]}\n"

        system = (
            f"You are analyzing data from '{self.domain.DOMAIN_NAME}'. "
            "Find patterns the statistical detectors might miss.\n\n"
            "For each pattern, output EXACTLY this format (blocks separated by ---):\n\n"
            "PATTERN: a_snake_case_name\n"
            "CONCLUSION: one sentence describing the finding\n"
            "EVIDENCE: specific data points that support this\n"
            "CONFIDENCE: 0.0-1.0\n"
            "---\n\n"
            "If nothing interesting, output: NONE\n"
            "Be specific. Cite actual values. Max 4 patterns."
        )
        try:
            raw = self.llm(system, context)
            if raw:
                llm_conclusions = parse_conclusions(raw)
                new_conclusions.extend(llm_conclusions)
        except Exception as e:
            log.debug("Reflect LLM failed: %s", e)

        # Store conclusions (dedup + settle)
        now = self._now()
        for pattern, conclusion, confidence, evidence in new_conclusions:
            # Time-based dedup
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=120)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            exists = self.db.execute(
                "SELECT id FROM conclusions WHERE pattern = ? AND timestamp > ?",
                (pattern, cutoff),
            ).fetchone()
            if exists:
                continue

            # Semantic dedup
            recent_conc = self.db.execute(
                "SELECT conclusion FROM conclusions WHERE timestamp > ? "
                "ORDER BY id DESC LIMIT 40",
                ((datetime.now(timezone.utc) - timedelta(minutes=DEDUP_WINDOW_MINUTES)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),),
            ).fetchall()
            is_dup = any(
                word_overlap(conclusion, c[0]) >= DEDUP_OVERLAP_THRESHOLD
                for c in recent_conc
            )
            if is_dup:
                log.debug("Dedup skipped: %.60s", conclusion)
                continue

            self.db.execute(
                "INSERT INTO conclusions "
                "(timestamp, cycle, pattern, conclusion, confidence, evidence, evidence_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (now, self.cycle, pattern, conclusion, confidence, evidence),
            )

            # Settle logic: same pattern 2+ times at 0.85+ → settled
            same_pattern_count = self.db.execute(
                "SELECT COUNT(*) FROM conclusions "
                "WHERE pattern = ? AND confidence >= ?",
                (pattern, SETTLED_MIN_CONFIDENCE),
            ).fetchone()[0]
            if same_pattern_count >= SETTLED_THRESHOLD:
                self.db.execute(
                    "UPDATE conclusions SET settled = 1 WHERE pattern = ? AND settled = 0",
                    (pattern,),
                )
                log.info("Settled: %s (appeared %d times at %.2f+)",
                         pattern, same_pattern_count, SETTLED_MIN_CONFIDENCE)

        self.db.commit()
        if new_conclusions:
            log.info("Reflect [%d]: %d conclusions", self.cycle, len(new_conclusions))

    # ── Grow ────────────────────────────────────────────────────────────

    def _grow(self):
        """Generate new questions, retire stale ones."""
        # Gather context for LLM question generation
        recent_conclusions = self.db.execute(
            "SELECT pattern, conclusion FROM conclusions ORDER BY id DESC LIMIT 5"
        ).fetchall()
        recent_thoughts = self.db.execute(
            "SELECT question, answer FROM thoughts ORDER BY id DESC LIMIT 10"
        ).fetchall()
        existing_qs = self.db.execute(
            "SELECT question FROM questions WHERE retired = 0"
        ).fetchall()
        existing_set = {q[0].lower() for q in existing_qs}

        new_questions: list[tuple[str, str]] = []

        if recent_conclusions or recent_thoughts:
            prompt = f"You are watching a system called '{self.domain.DOMAIN_NAME}'.\n\n"
            if recent_conclusions:
                prompt += "Conclusions formed so far:\n"
                for _, conclusion in recent_conclusions:
                    prompt += f"  - {conclusion}\n"
            if recent_thoughts:
                prompt += "\nRecent thoughts:\n"
                for q, a in recent_thoughts[:5]:
                    prompt += f"  Q: {q}\n  A: {a[:150]}\n"
            prompt += "\nExisting questions:\n"
            for q in list(existing_set)[:10]:
                prompt += f"  - {q}\n"
            prompt += (
                "\nGenerate 1-3 NEW questions I should be asking about this system. "
                "Questions that would help me understand it better. "
                "One question per line. Just the questions."
            )
            system = (
                f"You are an autonomous oversight system watching '{self.domain.DOMAIN_NAME}'. "
                "Generate insightful questions."
            )
            try:
                response = self.llm(system, prompt)
                if response:
                    for line in response.strip().split("\n"):
                        q = line.strip().lstrip("0123456789.-) ").strip()
                        if q and len(q) > 5 and q.endswith("?") and q.lower() not in existing_set:
                            new_questions.append((q, "self-generated"))
                            existing_set.add(q.lower())
            except Exception as e:
                log.debug("Growth LLM failed: %s", e)

        # Store new questions
        now = self._now()
        for q, origin in new_questions:
            self.db.execute(
                "INSERT OR IGNORE INTO questions (question, origin, created) VALUES (?, ?, ?)",
                (q, origin, now),
            )
        if new_questions:
            self.db.commit()
            log.info("Grow [%d]: %d new questions", self.cycle, len(new_questions))

        # Retire over-asked questions
        stale = self.db.execute(
            "SELECT id, question FROM questions WHERE times_asked > ? AND retired = 0",
            (QUESTION_RETIRE_THRESHOLD,),
        ).fetchall()
        for qid, q in stale:
            self.db.execute("UPDATE questions SET retired = 1 WHERE id = ?", (qid,))
            log.info("Retired (over-asked): \"%.60s\"", q)

        # Stagnation retirement: asked N+ times but never hit 0.50
        stagnant = self.db.execute(
            "SELECT q.id, q.question FROM questions q "
            "WHERE q.retired = 0 AND q.times_asked >= ? "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM thoughts t "
            "  WHERE t.question = q.question AND t.confidence >= 0.50"
            ")",
            (STAGNATION_MIN_ASKS,),
        ).fetchall()
        for qid, q in stagnant:
            self.db.execute("UPDATE questions SET retired = 1 WHERE id = ?", (qid,))
            log.info("Retired (stagnant): \"%.60s\"", q)

        # Hard cap on active questions
        active_count = self.db.execute(
            "SELECT COUNT(*) FROM questions WHERE retired = 0"
        ).fetchone()[0]
        if active_count > MAX_ACTIVE_QUESTIONS:
            overflow = active_count - MAX_ACTIVE_QUESTIONS
            to_prune = self.db.execute(
                "SELECT id FROM questions WHERE retired = 0 "
                "ORDER BY times_asked ASC, id ASC LIMIT ?",
                (overflow,),
            ).fetchall()
            for (qid,) in to_prune:
                self.db.execute("UPDATE questions SET retired = 1 WHERE id = ?", (qid,))
            log.info("Pruned %d questions (cap=%d)", len(to_prune), MAX_ACTIVE_QUESTIONS)

        if stale or stagnant:
            self.db.commit()

    # ── Prune ───────────────────────────────────────────────────────────

    def _prune(self):
        """Delete old observations and thoughts."""
        obs_cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=OBSERVATIONS_RETAIN_HOURS)
        ).strftime("%Y-%m-%d %H:%M:%S")
        thought_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=THOUGHTS_RETAIN_DAYS)
        ).strftime("%Y-%m-%d %H:%M:%S")

        self.db.execute("DELETE FROM observations WHERE timestamp < ?", (obs_cutoff,))
        self.db.execute("DELETE FROM thoughts WHERE timestamp < ?", (thought_cutoff,))
        self.db.commit()
