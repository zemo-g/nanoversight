"""nanoversight configuration — raw constants only."""

# ── Cycle timing ────────────────────────────────────────────────────────
CYCLE_INTERVAL = 60          # seconds between cycles
REFLECT_EVERY = 5            # cycles between reflect passes
GROW_EVERY = 15              # cycles between question growth

# ── Retention ───────────────────────────────────────────────────────────
OBSERVATIONS_RETAIN_HOURS = 72
THOUGHTS_RETAIN_DAYS = 30

# ── Questions ───────────────────────────────────────────────────────────
MAX_ACTIVE_QUESTIONS = 100
QUESTION_RETIRE_THRESHOLD = 5       # retire after N asks with no progress
STAGNATION_MIN_ASKS = 3             # retire if asked N+ times and never hit 0.50

# ── Conclusions ─────────────────────────────────────────────────────────
SETTLED_THRESHOLD = 2               # same pattern N times at high conf → settled
SETTLED_MIN_CONFIDENCE = 0.85
MAX_SETTLED = 20                    # cap settled conclusions in context

# ── Confidence ──────────────────────────────────────────────────────────
REPETITION_OVERLAP_THRESHOLD = 0.6  # word overlap > this → cap confidence
REPETITION_CONFIDENCE_CAP = 0.35
DEDUP_WINDOW_MINUTES = 60           # semantic dedup window for conclusions
DEDUP_OVERLAP_THRESHOLD = 0.62

# ── LLM defaults ───────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048
