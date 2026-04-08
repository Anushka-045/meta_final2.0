"""
Typed Pydantic models for the Compliance Monitor OpenEnv environment.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import uuid


# ─── Domain models ────────────────────────────────────────────────────────────

class ComplianceRule(BaseModel):
    id: str
    category: str
    severity_hint: Literal["Low", "Medium", "High", "Critical"]
    text: str
    applies_to: str  # "employee" | "contract" | "transaction"
    field: Optional[str] = None
    condition: Optional[str] = None
    conflicts_with: Optional[str] = None


class CompanyRecord(BaseModel):
    id: str
    type: Literal["employee", "contract", "transaction"]
    data: Dict[str, Any]


class Violation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    record_id: str
    rule_id: str
    reason: str = ""
    severity: Optional[Literal["Low", "Medium", "High", "Critical"]] = None
    explanation: str = ""
    fix: str = ""
    resolved: bool = False


class PolicyConflict(BaseModel):
    rule_id_a: str
    rule_id_b: str
    resolution: str = ""


# ─── Action models ────────────────────────────────────────────────────────────

class CheckRecordAction(BaseModel):
    action: Literal["check_record"]
    record_id: str


class FlagViolationAction(BaseModel):
    action: Literal["flag_violation"]
    record_id: str
    rule_id: str
    reason: str


class AssignSeverityAction(BaseModel):
    action: Literal["assign_severity"]
    violation_id: str
    severity: Literal["Low", "Medium", "High", "Critical"]


class GenerateExplanationAction(BaseModel):
    action: Literal["generate_explanation"]
    violation_id: str
    explanation: str


class SuggestFixAction(BaseModel):
    action: Literal["suggest_fix"]
    violation_id: str
    fix: str


class ResolveConflictAction(BaseModel):
    action: Literal["resolve_conflict"]
    rule_id_a: str
    rule_id_b: str
    resolution: str


# ─── Environment I/O models ───────────────────────────────────────────────────

class EnvState(BaseModel):
    records: List[Dict[str, Any]]
    rules: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    checked_record_ids: List[str]
    current_record_index: int
    episode_step: int
    max_steps: int
    done: bool
    total_reward: float
    task_id: str


class StepResult(BaseModel):
    observation: EnvState
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: Literal["task_easy", "task_medium", "task_hard"] = "task_easy"
    seed: Optional[int] = 42


class ResetResponse(BaseModel):
    observation: EnvState
    info: Dict[str, Any]


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    max_steps: int
    description: str
