"""
Core Compliance Monitor Environment.
Implements reset() / step() / state() following OpenEnv spec.
"""
import copy, uuid
from typing import Any, Dict, Optional, List

from models import (
    EnvState, StepResult, Violation, PolicyConflict,
    CheckRecordAction, FlagViolationAction, AssignSeverityAction,
    GenerateExplanationAction, SuggestFixAction, ResolveConflictAction,
)
from synthetic_data import (
    ALL_RECORDS, COMPLIANCE_RULES, CONFLICTING_RULES, GROUND_TRUTH_VIOLATIONS
)

TASK_CONFIGS = {
    "task_easy": {
        "name": "Single Record vs Single Rule",
        "description": "Agent checks one employee record against one compliance rule.",
        "difficulty": "easy",
        "max_steps": 10,
        "record_slice": slice(0, 1),
        "rule_slice": slice(0, 1),
        "include_conflicts": False,
    },
    "task_medium": {
        "name": "Multi-Record Multi-Rule Compliance Scan",
        "description": "Agent scans 10 records against 5 rules with partial credit.",
        "difficulty": "medium",
        "max_steps": 60,
        "record_slice": slice(0, 10),
        "rule_slice": slice(0, 5),
        "include_conflicts": False,
    },
    "task_hard": {
        "name": "Full DB Scan with Conflicting Policies",
        "description": "Agent scans all 30+ records against 10 rules including 2 contradicting ones.",
        "difficulty": "hard",
        "max_steps": 200,
        "record_slice": slice(0, None),
        "rule_slice": slice(0, None),
        "include_conflicts": True,
    },
}

EU_COUNTRIES = {"DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL", "SE", "FI"}
GDPR_REGIONS = EU_COUNTRIES | {"UK"}


class ComplianceEnvironment:
    def __init__(self):
        self._state: Optional[EnvState] = None
        self._task_id: str = "task_easy"
        self._violations: Dict[str, Violation] = {}
        self._conflicts: List[PolicyConflict] = []
        self._checked_ids: set = set()

    # ─── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_easy", seed: int = 42) -> Dict:
        cfg = TASK_CONFIGS[task_id]
        self._task_id = task_id
        self._violations = {}
        self._conflicts = []
        self._checked_ids = set()

        records = ALL_RECORDS[cfg["record_slice"]]
        rules = COMPLIANCE_RULES[cfg["rule_slice"]]
        if cfg["include_conflicts"]:
            rules = rules + CONFLICTING_RULES

        self._records = {r["id"]: r for r in records}
        self._rules = {r["id"]: r for r in rules}

        self._state = self._build_state(step=0, done=False, total_reward=0.0,
                                        max_steps=cfg["max_steps"], task_id=task_id)
        return {
            "observation": self._state.model_dump(),
            "info": {"task": cfg["name"], "difficulty": cfg["difficulty"],
                     "num_records": len(records), "num_rules": len(rules)}
        }

    def step(self, action: Dict[str, Any]) -> Dict:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            return {"observation": self._state.model_dump(), "reward": 0.0,
                    "done": True, "info": {"message": "Episode already done"}}

        reward, info = self._dispatch(action)
        step = self._state.episode_step + 1
        max_steps = TASK_CONFIGS[self._task_id]["max_steps"]
        total_reward = round(self._state.total_reward + reward, 4)

        # Episode ends when all records checked or max steps reached
        done = (len(self._checked_ids) >= len(self._records)) or (step >= max_steps)
        self._state = self._build_state(step=step, done=done,
                                        total_reward=total_reward, max_steps=max_steps,
                                        task_id=self._task_id)
        return {
            "observation": self._state.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }

    def state(self) -> Dict:
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state.model_dump()

    # ─── Action dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, action: Dict) -> tuple[float, Dict]:
        act = action.get("action")
        if act == "check_record":
            return self._check_record(action)
        elif act == "flag_violation":
            return self._flag_violation(action)
        elif act == "assign_severity":
            return self._assign_severity(action)
        elif act == "generate_explanation":
            return self._generate_explanation(action)
        elif act == "suggest_fix":
            return self._suggest_fix(action)
        elif act == "resolve_conflict":
            return self._resolve_conflict(action)
        else:
            return 0.0, {"error": f"Unknown action: {act}"}

    def _check_record(self, action: Dict) -> tuple[float, Dict]:
        record_id = action.get("record_id", "")
        if record_id not in self._records:
            return 0.0, {"error": f"Record {record_id} not found"}
        self._checked_ids.add(record_id)
        record = self._records[record_id]
        # Auto-detect if any rules apply to this record type
        applicable_rules = [r for r in self._rules.values()
                            if r.get("applies_to") == record.get("type")]
        return 0.0, {"record": record, "applicable_rules": [r["id"] for r in applicable_rules],
                     "message": f"Checked {record_id}, {len(applicable_rules)} applicable rules"}

    def _flag_violation(self, action: Dict) -> tuple[float, Dict]:
        record_id = action.get("record_id", "")
        rule_id = action.get("rule_id", "")
        reason = action.get("reason", "")

        if record_id not in self._records:
            return 0.0, {"error": f"Record {record_id} not found"}
        if rule_id not in self._rules:
            return 0.0, {"error": f"Rule {rule_id} not found"}

        # Check if this is a true positive
        is_true_positive = any(
            v["record_id"] == record_id and v["rule_id"] == rule_id
            for v in GROUND_TRUTH_VIOLATIONS
        )
        # Check for false positive (record not in ground truth for this rule)
        if not is_true_positive:
            return -0.1, {"message": f"False positive: {record_id} x {rule_id} not a real violation",
                          "penalty": -0.1}

        # Check duplicate
        existing = [v for v in self._violations.values()
                    if v.record_id == record_id and v.rule_id == rule_id]
        if existing:
            return 0.0, {"message": "Violation already flagged", "violation_id": existing[0].id}

        vid = str(uuid.uuid4())[:8]
        v = Violation(id=vid, record_id=record_id, rule_id=rule_id, reason=reason)
        self._violations[vid] = v
        return 0.4, {"message": "Violation flagged ✓", "violation_id": vid, "reward": 0.4}

    def _assign_severity(self, action: Dict) -> tuple[float, Dict]:
        vid = action.get("violation_id", "")
        severity = action.get("severity", "")
        if vid not in self._violations:
            return 0.0, {"error": f"Violation {vid} not found"}

        v = self._violations[vid]
        gt = next((g for g in GROUND_TRUTH_VIOLATIONS
                   if g["record_id"] == v.record_id and g["rule_id"] == v.rule_id), None)
        v.severity = severity

        if gt and gt["expected_severity"] == severity:
            return 0.2, {"message": "Severity correct ✓", "reward": 0.2}
        elif gt:
            # Partial credit for adjacent severity
            severity_order = ["Low", "Medium", "High", "Critical"]
            expected_idx = severity_order.index(gt["expected_severity"])
            actual_idx = severity_order.index(severity)
            if abs(expected_idx - actual_idx) == 1:
                return 0.1, {"message": "Severity off by one level", "reward": 0.1,
                              "expected": gt["expected_severity"]}
            return 0.0, {"message": f"Wrong severity. Expected {gt['expected_severity']}", "reward": 0.0}
        return 0.1, {"message": "Severity recorded (no ground truth reference)", "reward": 0.1}

    def _generate_explanation(self, action: Dict) -> tuple[float, Dict]:
        vid = action.get("violation_id", "")
        explanation = action.get("explanation", "")
        if vid not in self._violations:
            return 0.0, {"error": f"Violation {vid} not found"}

        v = self._violations[vid]
        # Quality heuristic: explanation must mention record_id, rule concept, and be >30 chars
        rule = self._rules.get(v.rule_id, {})
        rule_keywords = set(rule.get("text", "").lower().split())

        explanation_words = set(explanation.lower().split())
        overlap = len(rule_keywords & explanation_words)
        quality = min(1.0, overlap / max(len(rule_keywords) * 0.3, 1))

        v.explanation = explanation
        reward = round(0.2 * quality, 3)
        return reward, {"message": "Explanation recorded", "quality_score": quality, "reward": reward}

    def _suggest_fix(self, action: Dict) -> tuple[float, Dict]:
        vid = action.get("violation_id", "")
        fix = action.get("fix", "")
        if vid not in self._violations:
            return 0.0, {"error": f"Violation {vid} not found"}

        v = self._violations[vid]
        # Quality heuristic: fix should be actionable (>20 chars, not generic)
        generic_phrases = {"fix this", "resolve it", "update record", "check again"}
        is_generic = fix.lower().strip() in generic_phrases
        has_length = len(fix) > 20

        quality = 0.0 if is_generic else (1.0 if has_length else 0.5)
        v.fix = fix
        reward = round(0.2 * quality, 3)
        return reward, {"message": "Fix suggestion recorded", "quality_score": quality, "reward": reward}

    def _resolve_conflict(self, action: Dict) -> tuple[float, Dict]:
        rule_id_a = action.get("rule_id_a", "")
        rule_id_b = action.get("rule_id_b", "")
        resolution = action.get("resolution", "")

        if rule_id_a not in self._rules or rule_id_b not in self._rules:
            return 0.0, {"error": "One or both rule IDs not found"}

        # Check if this is a known conflict
        rule_a = self._rules[rule_id_a]
        rule_b = self._rules[rule_id_b]
        is_known_conflict = (
            rule_a.get("conflicts_with") == rule_id_b or
            rule_b.get("conflicts_with") == rule_id_a
        )

        conflict = PolicyConflict(rule_id_a=rule_id_a, rule_id_b=rule_id_b, resolution=resolution)
        self._conflicts.append(conflict)

        if is_known_conflict and len(resolution) > 30:
            return 0.3, {"message": "Known conflict resolved ✓", "reward": 0.3}
        elif is_known_conflict:
            return 0.15, {"message": "Known conflict found but resolution too brief", "reward": 0.15}
        else:
            return 0.05, {"message": "Conflict recorded (not in known conflict list)", "reward": 0.05}

    # ─── State builder ─────────────────────────────────────────────────────────

    def _build_state(self, step: int, done: bool, total_reward: float,
                     max_steps: int, task_id: str) -> EnvState:
        return EnvState(
            records=list(self._records.values()),
            rules=list(self._rules.values()),
            violations=[v.model_dump() for v in self._violations.values()],
            conflicts=[c.model_dump() for c in self._conflicts],
            checked_record_ids=list(self._checked_ids),
            current_record_index=len(self._checked_ids),
            episode_step=step,
            max_steps=max_steps,
            done=done,
            total_reward=total_reward,
            task_id=task_id,
        )
