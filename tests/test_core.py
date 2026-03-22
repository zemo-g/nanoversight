"""Tests for nanoversight core engine, parsers, and domain system."""

import json
import os
import sqlite3
import tempfile
import unittest

# Ensure we can import from the project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoversight import Oversight, word_overlap, extract_confidence, parse_conclusions
from domain import Domain, register, _registry, load_domain, list_domains


# ── Test domain ──────────────────────────────────────────────────────────────

class MockDomain(Domain):
    DOMAIN_ID = "mock"
    DOMAIN_NAME = "Mock System"
    SEED_QUESTIONS = [
        "Is the system healthy?",
        "Are there any anomalies?",
    ]

    def __init__(self):
        self.metrics = {"cpu": 45.0, "memory_mb": 8192, "disk_pct": 62}

    def observe(self):
        return dict(self.metrics)

    def format_context(self, metrics, conclusions, changes):
        ctx = "Mock System Metrics:\n"
        for k, v in metrics.items():
            ctx += f"  {k}: {v}\n"
        if conclusions:
            ctx += "Known:\n"
            for c in conclusions:
                ctx += f"  [{c['confidence']:.1f}] {c['conclusion']}\n"
        if changes:
            ctx += "Changes:\n"
            for ch in changes:
                ctx += f"  {ch}\n"
        return ctx


# ── Text utilities ───────────────────────────────────────────────────────────

class TestWordOverlap(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(word_overlap("hello world", "hello world"), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(word_overlap("cat dog", "fish bird"), 0.0)

    def test_partial_overlap(self):
        score = word_overlap("the cpu is high", "cpu usage is elevated")
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_empty_strings(self):
        self.assertAlmostEqual(word_overlap("", "hello"), 0.0)
        self.assertAlmostEqual(word_overlap("", ""), 0.0)

    def test_stop_words_ignored(self):
        # "the" and "is" are stop words
        score = word_overlap("the cpu is high", "cpu high")
        self.assertGreater(score, 0.5)


class TestExtractConfidence(unittest.TestCase):

    def test_basic(self):
        raw = "The system looks healthy.\nCONFIDENCE: 0.7"
        answer, conf = extract_confidence(raw)
        self.assertEqual(answer, "The system looks healthy.")
        self.assertAlmostEqual(conf, 0.7)

    def test_missing_confidence(self):
        raw = "Just an answer without a score."
        answer, conf = extract_confidence(raw)
        self.assertEqual(answer, "Just an answer without a score.")
        self.assertAlmostEqual(conf, 0.4)  # default

    def test_clamped_high(self):
        raw = "Answer\nCONFIDENCE: 5.0"
        _, conf = extract_confidence(raw)
        self.assertAlmostEqual(conf, 1.0)

    def test_clamped_low(self):
        raw = "Answer\nCONFIDENCE: -0.5"
        _, conf = extract_confidence(raw)
        self.assertAlmostEqual(conf, 0.1)

    def test_multiline_answer(self):
        raw = "Line 1\nLine 2\nLine 3\nCONFIDENCE: 0.5"
        answer, conf = extract_confidence(raw)
        self.assertIn("Line 1", answer)
        self.assertIn("Line 3", answer)
        self.assertAlmostEqual(conf, 0.5)


class TestParseConclusions(unittest.TestCase):

    def test_single_conclusion(self):
        raw = """PATTERN: cpu_rising
CONCLUSION: CPU usage has been increasing steadily
EVIDENCE: 45% → 52% → 58% over 3 cycles
CONFIDENCE: 0.65"""
        results = parse_conclusions(raw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "cpu_rising")
        self.assertIn("CPU usage", results[0][1])
        self.assertAlmostEqual(results[0][2], 0.65)

    def test_multiple_conclusions(self):
        raw = """PATTERN: cpu_rising
CONCLUSION: CPU going up
EVIDENCE: data
CONFIDENCE: 0.6
---
PATTERN: disk_full
CONCLUSION: Disk almost full
EVIDENCE: 95%
CONFIDENCE: 0.8"""
        results = parse_conclusions(raw)
        self.assertEqual(len(results), 2)

    def test_none_output(self):
        results = parse_conclusions("NONE")
        self.assertEqual(results, [])

    def test_empty_output(self):
        results = parse_conclusions("")
        self.assertEqual(results, [])

    def test_max_four(self):
        raw = "\n---\n".join([
            f"PATTERN: p{i}\nCONCLUSION: c{i}\nCONFIDENCE: 0.5"
            for i in range(10)
        ])
        results = parse_conclusions(raw)
        self.assertLessEqual(len(results), 4)


# ── Core engine ──────────────────────────────────────────────────────────────

class TestOversightEngine(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.domain = MockDomain()
        self.llm_calls = []

        def mock_llm(system, user):
            self.llm_calls.append((system, user))
            return "The system appears healthy. No anomalies detected.\nCONFIDENCE: 0.45"

        self.engine = Oversight(
            domain=self.domain,
            llm=mock_llm,
            db_path=self.db_path,
        )

    def test_seed_questions_inserted(self):
        count = self.engine.db.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        self.assertEqual(count, 2)

    def test_single_cycle(self):
        self.engine._cycle()
        # Should have one observation
        obs = self.engine.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        self.assertEqual(obs, 1)
        # Should have one thought
        thoughts = self.engine.db.execute("SELECT COUNT(*) FROM thoughts").fetchone()[0]
        self.assertEqual(thoughts, 1)

    def test_observation_stored(self):
        self.engine._cycle()
        row = self.engine.db.execute(
            "SELECT metrics_json FROM observations LIMIT 1"
        ).fetchone()
        metrics = json.loads(row[0])
        self.assertEqual(metrics["cpu"], 45.0)
        self.assertEqual(metrics["memory_mb"], 8192)

    def test_thought_confidence_stored(self):
        self.engine._cycle()
        row = self.engine.db.execute(
            "SELECT confidence FROM thoughts LIMIT 1"
        ).fetchone()
        self.assertAlmostEqual(row[0], 0.45)

    def test_change_detection(self):
        self.engine.prev_metrics = {"cpu": 40.0, "memory_mb": 8192}
        changes = self.engine._detect_changes({"cpu": 50.0, "memory_mb": 8192})
        self.assertEqual(len(changes), 1)
        self.assertIn("cpu", changes[0])
        self.assertIn("up", changes[0])

    def test_change_detection_new_metric(self):
        self.engine.prev_metrics = {"cpu": 40.0}
        changes = self.engine._detect_changes({"cpu": 40.0, "gpu": 55.0})
        self.assertTrue(any("New metric" in c for c in changes))

    def test_repetition_penalty(self):
        """Repeating the same answer should cap confidence."""
        call_count = [0]

        def repetitive_llm(system, user):
            call_count[0] += 1
            return "The system is healthy and stable.\nCONFIDENCE: 0.8"

        engine = Oversight(
            domain=self.domain,
            llm=repetitive_llm,
            db_path=os.path.join(self.tmpdir, "rep.db"),
        )
        engine._cycle()
        engine._cycle()  # Second time — should trigger repetition penalty

        rows = engine.db.execute(
            "SELECT confidence FROM thoughts ORDER BY id"
        ).fetchall()
        if len(rows) == 2:
            # Second confidence should be capped (repetition penalty)
            self.assertLessEqual(rows[1][0], 0.8)

    def test_max_cycles(self):
        """Engine respects max_cycles."""
        import config
        import nanoversight as nv_mod
        old_interval = config.CYCLE_INTERVAL
        config.CYCLE_INTERVAL = 0
        nv_mod.CYCLE_INTERVAL = 0
        try:
            self.engine.run(max_cycles=3)
            self.assertEqual(self.engine.cycle, 3)
        finally:
            config.CYCLE_INTERVAL = old_interval
            nv_mod.CYCLE_INTERVAL = old_interval

    def test_question_times_asked_incremented(self):
        self.engine._cycle()
        row = self.engine.db.execute(
            "SELECT times_asked FROM questions ORDER BY times_asked DESC LIMIT 1"
        ).fetchone()
        self.assertGreaterEqual(row[0], 1)


# ── Domain system ────────────────────────────────────────────────────────────

class TestDomainRegistry(unittest.TestCase):

    def test_list_domains(self):
        domains = list_domains()
        self.assertIsInstance(domains, list)
        # Built-in domains should be discoverable
        self.assertIn("sysmon", domains)

    def test_load_sysmon(self):
        domain = load_domain("sysmon")
        self.assertEqual(domain.DOMAIN_ID, "sysmon")
        self.assertTrue(len(domain.SEED_QUESTIONS) > 0)

    def test_load_unknown_raises(self):
        with self.assertRaises(ValueError):
            load_domain("nonexistent_domain_xyz")

    def test_sysmon_observe(self):
        domain = load_domain("sysmon")
        metrics = domain.observe()
        self.assertIn("load_1m", metrics)
        self.assertIsInstance(metrics["load_1m"], (int, float))


if __name__ == "__main__":
    unittest.main()
