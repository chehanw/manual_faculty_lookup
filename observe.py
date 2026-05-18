"""
observe.py — Observability module for the AI Faculty Pipeline.

Provides:
  - Structured JSON logging (one JSONL file per run)
  - Thread-safe in-memory metrics collection
  - LLM cost estimation
  - Persistent run history in SQLite (pipeline_runs table)
  - Alert checks for coverage, error rate, and cost

Usage:
    import observe
    observe.start("20260305_102345_a3f8b1")
    observe.log("pipeline_start", discipline="Pediatric Endocrinology")
    observe.metric("faculty_enriched")
    observe.add_cost("openai_responses_web_search")
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────
# RUN STATE
# ─────────────────────────────────────────────

RUN_ID: str = ""
_log_file: str = ""
_logger: logging.Logger | None = None
_logger_lock = threading.Lock()

# ─────────────────────────────────────────────
# COST ESTIMATES (gpt-4o, 2026)
# ─────────────────────────────────────────────

COST_PER_CALL = {
    "openai_responses_web_search": 0.025,   # flat per web search call
    "openai_chat_input_per_1k":    0.0025,  # $2.50 / 1M input tokens
    "openai_chat_output_per_1k":   0.010,   # $10.00 / 1M output tokens
}

# ─────────────────────────────────────────────
# ALERT THRESHOLDS
# ─────────────────────────────────────────────

ALERT_THRESHOLDS = {
    "min_faculty_expected": 10,
    "max_error_rate":       0.10,
    "max_cost_usd":         5.00,
    "min_nih_coverage":     0.30,
}

# ─────────────────────────────────────────────
# METRICS (thread-safe)
# ─────────────────────────────────────────────

_metrics_lock = threading.Lock()
_metrics: dict = {}

def _reset_metrics():
    global _metrics
    _metrics = {
        "faculty_enriched":   0,
        "cache_hits_nih":     0,
        "cache_misses_nih":   0,
        "cache_hits_roles":   0,
        "cache_misses_roles": 0,
        "api_calls": {
            "nih": 0, "scopus": 0,
            "openai_responses": 0, "openai_chat": 0
        },
        "api_errors": {
            "nih": 0, "scopus": 0, "openai": 0, "scraping": 0
        },
        "llm_cost_usd":  0.0,
        "latencies_ms":  [],
        "field_coverage": {
            "nih_grants": 0, "editorial": 0, "society": 0,
            "training": 0, "leadership": 0, "h_index": 0
        },
    }


def metric(key: str, value: float = 1, sub: str | None = None):
    """Increment a metric counter. Thread-safe."""
    with _metrics_lock:
        if sub:
            _metrics[key][sub] += value
        elif isinstance(_metrics.get(key), list):
            _metrics[key].append(value)
        else:
            _metrics[key] += value


def get_metrics() -> dict:
    """Return a snapshot of current metrics with derived fields."""
    with _metrics_lock:
        m = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
             for k, v in _metrics.items()}

    lats = m["latencies_ms"]
    m["avg_latency_ms"] = round(sum(lats) / len(lats), 1) if lats else 0
    m["max_latency_ms"] = max(lats) if lats else 0

    nih_total   = m["cache_hits_nih"]   + m["cache_misses_nih"]
    roles_total = m["cache_hits_roles"] + m["cache_misses_roles"]
    m["cache_hit_rate_nih"]   = round(m["cache_hits_nih"]   / nih_total,   3) if nih_total   else 0.0
    m["cache_hit_rate_roles"] = round(m["cache_hits_roles"] / roles_total, 3) if roles_total else 0.0
    return m

# ─────────────────────────────────────────────
# COST TRACKING
# ─────────────────────────────────────────────

def add_cost(call_type: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Accumulate estimated LLM cost and return the amount added."""
    cost = COST_PER_CALL.get(call_type, 0.0)
    cost += (input_tokens  / 1000) * COST_PER_CALL["openai_chat_input_per_1k"]
    cost += (output_tokens / 1000) * COST_PER_CALL["openai_chat_output_per_1k"]
    with _metrics_lock:
        _metrics["llm_cost_usd"] += cost
    return cost

# ─────────────────────────────────────────────
# STRUCTURED LOGGING
# ─────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "event":     record.getMessage(),
            "run_id":    RUN_ID,
        }
        entry.update(getattr(record, "_extra", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _get_logger() -> logging.Logger:
    global _logger
    with _logger_lock:
        if _logger is None:
            Path("logs").mkdir(exist_ok=True)
            logger = logging.getLogger(f"pipeline.{RUN_ID}")
            logger.propagate = False
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(_log_file, encoding="utf-8")
            fh.setFormatter(_JsonFormatter())
            logger.addHandler(fh)

            _logger = logger
    return _logger


def log(event: str, level: str = "info", **kwargs):
    """Emit a structured JSON log event."""
    record = logging.LogRecord(
        name="pipeline", level=getattr(logging, level.upper(), logging.INFO),
        pathname="", lineno=0, msg=event, args=(), exc_info=None,
    )
    record._extra = kwargs
    _get_logger().handle(record)

# ─────────────────────────────────────────────
# RUN TRACKING (SQLite)
# ─────────────────────────────────────────────

def init_run_tracking(db_path: str):
    """Create pipeline_runs table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL,
            discipline      TEXT,
            schools         TEXT,
            faculty_count   INTEGER,
            runtime_seconds REAL,
            cache_hit_rate  REAL,
            llm_cost_usd    REAL,
            api_errors      INTEGER,
            success_flag    INTEGER,
            log_file        TEXT
        )
    """)
    conn.commit()
    conn.close()


def record_run(db_path: str, discipline: str, schools: list,
               faculty_count: int, runtime: float, success: bool):
    """Write one row to pipeline_runs at the end of a run."""
    m = get_metrics()
    cache_hit_rate = (m["cache_hit_rate_nih"] + m["cache_hit_rate_roles"]) / 2
    total_errors   = sum(m["api_errors"].values())
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO pipeline_runs
          (run_id, timestamp, discipline, schools, faculty_count,
           runtime_seconds, cache_hit_rate, llm_cost_usd,
           api_errors, success_flag, log_file)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, [
        RUN_ID,
        datetime.now(timezone.utc).isoformat(),
        discipline,
        json.dumps(schools),
        faculty_count,
        round(runtime, 2),
        round(cache_hit_rate, 3),
        round(m["llm_cost_usd"], 4),
        total_errors,
        int(success),
        _log_file,
    ])
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# ALERT CHECKS
# ─────────────────────────────────────────────

def run_alert_checks(faculty_found: int, discipline: str) -> list[str]:
    """Check alert conditions. Logs warnings and returns list of triggered alerts."""
    m = get_metrics()
    alerts = []

    if faculty_found < ALERT_THRESHOLDS["min_faculty_expected"]:
        alerts.append(f"LOW_DISCOVERY: only {faculty_found} faculty found for {discipline}")

    total_api    = sum(m["api_calls"].values())
    total_errors = sum(m["api_errors"].values())
    if total_api and (total_errors / total_api) > ALERT_THRESHOLDS["max_error_rate"]:
        alerts.append(f"HIGH_ERROR_RATE: {total_errors}/{total_api} API calls failed")

    if m["llm_cost_usd"] > ALERT_THRESHOLDS["max_cost_usd"]:
        alerts.append(f"COST_EXCEEDED: ${m['llm_cost_usd']:.2f} > ${ALERT_THRESHOLDS['max_cost_usd']:.2f}")

    nih_coverage = m["field_coverage"]["nih_grants"] / max(faculty_found, 1)
    if nih_coverage < ALERT_THRESHOLDS["min_nih_coverage"]:
        alerts.append(f"LOW_NIH_COVERAGE: {nih_coverage:.0%} (threshold {ALERT_THRESHOLDS['min_nih_coverage']:.0%})")

    for alert in alerts:
        log("alert", level="warning", alert=alert, discipline=discipline)

    return alerts

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

def start(run_id: str | None = None):
    """Initialize the observability system for a new run."""
    global RUN_ID, _log_file, _logger
    RUN_ID    = run_id or (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    _log_file = f"logs/pipeline_{RUN_ID}.jsonl"
    _logger   = None  # force new logger for this run
    _reset_metrics()
