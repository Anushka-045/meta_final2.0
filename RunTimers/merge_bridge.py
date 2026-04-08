"""
merge_bridge.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE SINGLE MERGE POINT BETWEEN RUNTIMERS TEAM.

Gargi imports this module inside ComplianceEnvironment.reset() to swap out
dummy data for RunTimers's real database + extracted rules.

USAGE — in Gargi's environment.py:
─────────────────────────────────────────────────────────────────────────────
from merge_bridge import load_real_data, MERGE_AVAILABLE

class ComplianceEnvironment:
    def reset(self, task_id="task_easy", seed=42):
        if MERGE_AVAILABLE:
            records, rules = load_real_data(task_id)
        else:
            records, rules = DUMMY_RECORDS, DUMMY_RULES   # Gargi's existing fallback
        ...
─────────────────────────────────────────────────────────────────────────────

The bridge is fully backward-compatible:
  - If RunTimers's server is unreachable, MERGE_AVAILABLE = False and Gargi's
    original dummy data is used automatically — no crash, no code change needed.
  - If the server IS reachable, real SQLite records and extracted rules are used.
"""

import os
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# Where RunTimers's server runs.
# Override via env var for HF Spaces: PIPELINE_URL=https://your-space.hf.space
PIPELINE_URL = os.environ.get("PIPELINE_URL", "http://localhost:7861")

# ── Check availability at import time ────────────────────────────────────────
try:
    import requests as _requests
    _r = _requests.get(f"{PIPELINE_URL}/health", timeout=3)
    MERGE_AVAILABLE = _r.status_code == 200
    if MERGE_AVAILABLE:
        logger.info(f"[merge_bridge] RunTimers pipeline reachable at {PIPELINE_URL}")
    else:
        logger.warning(f"[merge_bridge] Pipeline returned {_r.status_code} — using dummy data")
except Exception as _e:
    MERGE_AVAILABLE = False
    logger.warning(f"[merge_bridge] Pipeline unreachable ({_e}) — using dummy data")


# ── Task ID → record filter mapping ──────────────────────────────────────────
_TASK_FILTERS: Dict[str, Dict] = {
    "task_easy": {
        "record_ids": ["EMP001"],
        "rule_ids":   ["RULE001"],
    },
    "task_medium": {
        "record_ids": [f"EMP{i:03d}" for i in range(1, 31)],
        "rule_ids":   ["RULE001", "RULE002", "RULE003", "RULE004"],
    },
    "task_hard": {
        "record_ids": None,   # All records
        "rule_ids":   None,   # All rules
    },
}


def load_real_data(task_id: str = "task_easy") -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch records and rules from RunTimers's pipeline in the format
    Gargi's environment expects.

    Returns:
        (records, rules) — both are lists of dicts.

    Raises:
        RuntimeError if the pipeline is unreachable.
    """
    import requests

    filters = _TASK_FILTERS.get(task_id, _TASK_FILTERS["task_hard"])

    # ── Fetch records ──────────────────────────────────────────────────────
    resp_records = requests.get(f"{PIPELINE_URL}/openenv/records", timeout=10)
    resp_records.raise_for_status()
    all_records: List[Dict] = resp_records.json()["records"]

    # ── Fetch rules ────────────────────────────────────────────────────────
    resp_rules = requests.get(f"{PIPELINE_URL}/openenv/rules", timeout=10)
    resp_rules.raise_for_status()
    all_rules: List[Dict] = resp_rules.json()["rules"]

    # ── Apply task-level filtering ─────────────────────────────────────────
    record_ids = filters["record_ids"]
    rule_ids   = filters["rule_ids"]

    records = (
        [r for r in all_records if r.get("id") in record_ids]
        if record_ids is not None else all_records
    )
    rules = (
        [r for r in all_rules if r.get("id") in rule_ids]
        if rule_ids is not None else all_rules
    )

    logger.info(
        f"[merge_bridge] {task_id}: loaded {len(records)} records, {len(rules)} rules"
    )
    return records, rules


def get_pipeline_summary() -> Dict[str, Any]:
    """Return RunTimers's compliance summary (for display in Gargi's /state)."""
    import requests
    resp = requests.get(f"{PIPELINE_URL}/summary", timeout=5)
    resp.raise_for_status()
    return resp.json()


def run_pipeline_scan(task_id: str = "task_hard") -> List[Dict]:
    """
    Trigger a full pipeline scan and return violation list.
    Gargi can call this to pre-populate the episode with real violations.
    """
    import requests
    resp = requests.post(
        f"{PIPELINE_URL}/scan",
        json={"record_type": None, "include_explanations": False},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("violations", [])
