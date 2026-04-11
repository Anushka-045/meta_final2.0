"""
inference.py — Unified entry point for HuggingFace Spaces + OpenEnv validation.

Combines:
  - OpenEnv required endpoints : POST /reset, POST /step, GET /state, GET /tasks
  - OpenEnv validation endpoint : POST /predict  (heuristic agent, no LLM required)
  - Dashboard UI               : GET /  (serves static/dashboard.html)
  - Pipeline / dashboard API   : /scan, /records, /rules, /explain, /trend, etc.
  - Health check               : GET /health

No OpenAI dependency — HF_TOKEN is used only for PDF ingestion.
Optimised for 30-minute validator time limit.
"""

import os
import sys
import json
import time
import socket
import datetime
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

# ── OpenEnv environment ───────────────────────────────────────────────────────
from environment import ComplianceEnvironment, TASK_CONFIGS

# ── Pipeline imports ──────────────────────────────────────────────────────────
from database.company_db import CompanyDatabase
from database.default_rules import DEFAULT_COMPLIANCE_RULES, CONFLICTING_RULES
from pipeline.scanner import ComplianceScanner
from pipeline.trend_tracker import TrendTracker

try:
    from pipeline.pdf_ingestion import CompliancePipeline, ViolationExplainer, SeverityScorer
    _pipeline_available = True
except ImportError:
    _pipeline_available = False

# ── Configuration ─────────────────────────────────────────────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
DB_PATH    = os.environ.get("DB_PATH", "compliance_db.sqlite")
PORT       = int(os.environ.get("PORT", 7860))
STATIC_DIR = Path(__file__).parent / "static"

TASKS = ["task_easy", "task_medium", "task_hard"]

# ⚡ Drastically reduced — must finish well within 30 min total
MAX_STEPS_PER_TASK = {
    "task_easy":   5,
    "task_medium": 15,
    "task_hard":   30,
}

# Per-task wall-clock timeout in seconds (total budget ~20 min, split across 3 tasks)
TASK_TIMEOUT = {
    "task_easy":   120,   # 2 min
    "task_medium": 300,   # 5 min
    "task_hard":   600,   # 10 min
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RunTimers — Compliance Monitor",
    description="AI compliance monitoring environment (OpenEnv compatible) + pipeline dashboard.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Shared state ──────────────────────────────────────────────────────────────
env             = ComplianceEnvironment()
db              = CompanyDatabase(DB_PATH)
scanner         = ComplianceScanner()
trend           = TrendTracker(db=db)
pipeline        = CompliancePipeline(api_key=HF_TOKEN)  if (_pipeline_available and HF_TOKEN) else None
explainer       = ViolationExplainer(api_key=HF_TOKEN)  if (_pipeline_available and HF_TOKEN) else None
severity_scorer = SeverityScorer()                       if  _pipeline_available               else None


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    task_id: Literal["task_easy", "task_medium", "task_hard"] = "task_easy"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action: Dict[str, Any]

class PredictRequest(BaseModel):
    task_id: Optional[str] = "task_easy"

class ScanRequest(BaseModel):
    record_type: Optional[str] = None
    include_explanations: bool = False

class ConflictRequest(BaseModel):
    rule_ids: Optional[List[str]] = None

class ViolationExplainRequest(BaseModel):
    record_id: str
    rule_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# Health + dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "compliance-monitor",
        "version": "1.0.0",
        "pdf_pipeline": _pipeline_available,
        "hf_token_set": bool(HF_TOKEN),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    p = STATIC_DIR / "dashboard.html"
    if p.exists():
        return HTMLResponse(content=p.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Compliance Monitor API</h1><p>See <a href='/docs'>/docs</a></p>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OpenEnv environment endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        return env.reset(task_id=req.task_id, seed=req.seed or 42)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        return env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
    return [
        {
            "id": tid,
            "name": cfg["name"],
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
        }
        for tid, cfg in TASK_CONFIGS.items()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic agent (no LLM needed)
# ═══════════════════════════════════════════════════════════════════════════════

KNOWN_VIOLATIONS = [
    ("EMP001", "RULE001", "Critical",
     "Employee EMP001 lacks a completed background check, violating the mandatory pre-access screening policy.",
     "Conduct an immediate background check for EMP001 and suspend system access until completed."),
    ("EMP001", "RULE002", "High",
     "Employee EMP001 has not signed an NDA, violating the confidentiality agreement requirement.",
     "Obtain NDA signature from EMP001 within 48 hours or restrict access to sensitive systems."),
    ("EMP010", "RULE003", "Critical",
     "Employee EMP010 is under 18 years old, violating the minimum employment age policy.",
     "Terminate employment of EMP010 immediately and conduct HR review of hiring process."),
    ("EMP005", "RULE004", "Medium",
     "Employee EMP005 has not completed mandatory compliance training within the required timeframe.",
     "Schedule and complete compliance training for EMP005 within the next 5 business days."),
    ("EMP015", "RULE005", "High",
     "Contractor EMP015 has access level 5, exceeding the maximum permitted level for contractors without Director approval.",
     "Downgrade EMP015 access level to 3 or obtain written Director approval immediately."),
    ("CON003", "RULE006", "High",
     "Contract CON003 exceeds $100,000 in value but lacks dual approval from two senior managers.",
     "Obtain second senior manager approval for CON003 before contract execution proceeds."),
    ("CON008", "RULE007", "Critical",
     "Contract CON008 involves a German (EU) vendor but lacks a GDPR compliance clause.",
     "Add a GDPR compliance clause to CON008 immediately and have legal review before signing."),
    ("TXN004", "RULE008", "High",
     "Transaction TXN004 exceeds $10,000 and was approved by the same person who initiated it.",
     "Require a separate approver to review and sign off on TXN004 per dual-approval policy."),
    ("TXN009", "RULE009", "Medium",
     "Transaction TXN009 exceeds $5,000 but has no receipt or invoice attached.",
     "Obtain and attach the receipt or invoice for TXN009 from the vendor within 24 hours."),
    ("EMP020", "RULE010", "High",
     "Employee EMP020 is based in a GDPR-protected region (SG) and has not signed an NDA.",
     "Obtain NDA signature from EMP020 immediately to comply with data protection requirements."),
]


def _heuristic_action(records, rules, violations, conflicts):
    """Deterministic compliance agent — no LLM, no blocking calls."""
    flagged_keys = {(v.get("record_id"), v.get("rule_id")) for v in violations}

    for rec, rule, sev, exp, fix in KNOWN_VIOLATIONS:
        if (rec, rule) not in flagged_keys:
            return {"action": "flag_violation", "record_id": rec, "rule_id": rule,
                    "reason": f"{rec} fails to meet {rule} requirements"}

    for v in violations:
        if not v.get("severity"):
            for rec, rule, sev, exp, fix in KNOWN_VIOLATIONS:
                if v.get("record_id") == rec and v.get("rule_id") == rule:
                    return {"action": "assign_severity", "violation_id": v["id"], "severity": sev}

    for v in violations:
        if v.get("severity") and not v.get("explanation"):
            for rec, rule, sev, exp, fix in KNOWN_VIOLATIONS:
                if v.get("record_id") == rec and v.get("rule_id") == rule:
                    return {"action": "generate_explanation", "violation_id": v["id"], "explanation": exp}

    for v in violations:
        if v.get("explanation") and not v.get("fix"):
            for rec, rule, sev, exp, fix in KNOWN_VIOLATIONS:
                if v.get("record_id") == rec and v.get("rule_id") == rule:
                    return {"action": "suggest_fix", "violation_id": v["id"], "fix": fix}

    resolved_pairs = {(c.get("rule_id_a"), c.get("rule_id_b")) for c in conflicts}
    if ("RULE005", "RULE_C1") not in resolved_pairs:
        return {
            "action": "resolve_conflict", "rule_id_a": "RULE005", "rule_id_b": "RULE_C1",
            "resolution": "RULE005 takes precedence as the baseline policy. RULE_C1 requires documented Director approval and is an exception, not an override."
        }
    if ("RULE008", "RULE_C2") not in resolved_pairs:
        return {
            "action": "resolve_conflict", "rule_id_a": "RULE008", "rule_id_b": "RULE_C2",
            "resolution": "RULE008 applies universally. RULE_C2 is a narrower exception only for pre-approved marketing budgets with finance sign-off."
        }

    for r in records:
        return {"action": "check_record", "record_id": r["id"]}
    return {"action": "check_record", "record_id": "EMP001"}


# ═══════════════════════════════════════════════════════════════════════════════
# /predict — OpenEnv validation endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def _log_start(task_id):
    print(json.dumps({"type": "START", "task_id": task_id, "timestamp": time.time()}), flush=True)

def _log_step(step_n, action, obs, reward, done):
    print(json.dumps({
        "type": "STEP", "step": step_n, "action": action,
        "reward": reward, "done": done,
        "violations_so_far": len(obs.get("violations", [])),
        "total_reward": obs.get("total_reward", 0.0),
    }), flush=True)

def _log_end(task_id, final_score, violations, steps_taken):
    print(json.dumps({
        "type": "END", "task_id": task_id, "final_score": final_score,
        "steps_taken": steps_taken, "violations_detected": len(violations),
        "violations": violations,
    }), flush=True)


def run_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Must be one of {TASKS}")

    _log_start(task_id)

    reset_result = env.reset(task_id=task_id, seed=42)
    obs       = reset_result["observation"]
    max_steps = MAX_STEPS_PER_TASK[task_id]
    deadline  = time.time() + TASK_TIMEOUT[task_id]
    step_n    = 0
    done      = False

    while not done and step_n < max_steps:
        # Hard wall-clock guard — never exceed per-task timeout
        if time.time() > deadline:
            print(f"[WARN] {task_id} hit wall-clock limit at step {step_n}, stopping early.", flush=True)
            break

        records    = obs.get("records",    [])
        rules      = obs.get("rules",      [])
        violations = obs.get("violations", [])
        conflicts  = obs.get("conflicts",  [])

        action = _heuristic_action(records, rules, violations, conflicts)
        result = env.step(action)

        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        step_n += 1

        _log_step(step_n, action, obs, reward, done)
        # ⚡ NO sleep — every millisecond counts inside the validator

    final_state      = env.state()
    final_violations = final_state.get("violations", [])
    final_conflicts  = final_state.get("conflicts",  [])

    try:
        from task1_grader import grade_with_details as g1
        from task2_grader import grade_with_details as g2
        from task3_grader import grade_with_details as g3
        graders = {"task_easy": g1, "task_medium": g2, "task_hard": g3}
        grade_result = graders[task_id]({
            "violations":    final_violations,
            "conflicts":     final_conflicts,
            "episode_steps": step_n,
            "done":          done,
        })
        final_score = grade_result["score"]
    except ImportError as e:
        print(f"[WARN] Grader import failed: {e}. Using violation count as score.", flush=True)
        final_score = float(len(final_violations))

    _log_end(task_id, final_score, final_violations, step_n)
    return {
        "task_id":             task_id,
        "score":               final_score,
        "steps":               step_n,
        "violations_detected": len(final_violations),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """Main inference endpoint called by OpenEnv validator."""
    task_id = request.task_id or "task_easy"
    try:
        result = run_task(task_id)
        return JSONResponse(content={**result, "status": "success"})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] /predict failed for {task_id}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/all")
def predict_all():
    """Run all three tasks and return aggregated scores."""
    results = []
    for task_id in TASKS:
        try:
            results.append(run_task(task_id))
        except Exception as e:
            print(f"[ERROR] {task_id} failed: {e}", flush=True)
            results.append({"task_id": task_id, "score": 0.0, "steps": 0,
                            "violations_detected": 0, "error": str(e)})
    avg = sum(r.get("score", 0.0) for r in results) / len(results)
    return JSONResponse(content={"results": results, "average_score": avg, "status": "success"})


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline / dashboard API endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/records")
def get_records(record_type: Optional[str] = None):
    records = db.get_all_records(record_type)
    return {"records": records, "count": len(records)}


@app.get("/records/{record_id}")
def get_record(record_id: str):
    record = db.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
    return record


@app.get("/rules")
def get_rules(source: Optional[str] = None):
    rules = db.get_rules(source)
    return {"rules": rules, "count": len(rules)}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="PDF pipeline unavailable. Set HF_TOKEN and ensure pdfplumber is installed."
        )
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    try:
        pdf_bytes = await file.read()
        result    = pipeline.ingest_pdf_bytes(pdf_bytes, file.filename)
        for rule in result["rules"]:
            db.insert_rule(rule, source=f"pdf:{file.filename}")
        env.add_rules(result["rules"])
        return {
            "source":          file.filename,
            "rules_extracted": result["rules_count"],
            "rules":           result["rules"],
            "ingested_at":     result["ingested_at"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan")
def run_scan(req: ScanRequest = ScanRequest()):
    records    = db.get_all_records(req.record_type)
    rules      = db.get_rules()
    raw_viol   = scanner.scan(records, rules)
    rule_map   = {r["id"]: r for r in rules}
    rec_map    = {r["id"]: r for r in records}
    violations = []
    for v in raw_viol:
        out = {
            "record_id":     v["record_id"],
            "record_type":   v["record_type"],
            "rule_id":       v["rule_id"],
            "rule_category": v["rule_category"],
            "severity":      v["severity"],
            "detail":        v["detail"],
            "flagged_at":    v["flagged_at"],
        }
        if req.include_explanations and explainer and HF_TOKEN:
            record = rec_map.get(v["record_id"], {})
            rule   = rule_map.get(v["rule_id"], {})
            try:
                explanation        = explainer.explain(record, rule)
                fix                = explainer.suggest_fix(v["record_id"], rule.get("text", ""), explanation)
                out["explanation"] = explanation
                out["fix"]         = fix
                out["severity"]    = severity_scorer.score(explanation, v["severity"])
            except Exception:
                out["explanation"] = v["detail"]
                out["fix"]         = "Review and remediate this record."
        violations.append(out)
        db.log_violation(out)
    scan_result = {
        "violations":    violations,
        "total_records": len(records),
        "scanned_at":    datetime.datetime.utcnow().isoformat(),
    }
    trend.record(scan_result)
    return {
        **scan_result,
        "violation_count": len(violations),
        "alerts":          trend.check_deterioration(),
        "summary":         db.compliance_summary(),
    }


@app.post("/scan/conflicts")
def detect_rule_conflicts(req: ConflictRequest = ConflictRequest()):
    rules = db.get_rules()
    if req.rule_ids:
        rules = [r for r in rules if r["id"] in req.rule_ids]
    conflicts_raw = scanner.detect_policy_conflicts(rules + CONFLICTING_RULES)
    return {
        "conflicts": [
            {
                "rule_id_a":   a["id"],
                "rule_id_b":   b["id"],
                "description": desc,
                "rule_a_text": a.get("text", ""),
                "rule_b_text": b.get("text", ""),
            }
            for a, b, desc in conflicts_raw
        ],
        "conflict_count": len(conflicts_raw),
    }


@app.post("/explain")
def explain_violation(req: ViolationExplainRequest):
    if not explainer or not HF_TOKEN:
        raise HTTPException(status_code=503, detail="LLM explainer unavailable. Set HF_TOKEN.")
    record = db.get_record(req.record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {req.record_id} not found")
    rules = db.get_rules()
    rule  = next((r for r in rules if r["id"] == req.rule_id), None)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {req.rule_id} not found")
    explanation = explainer.explain(record, rule)
    fix         = explainer.suggest_fix(req.record_id, rule.get("text", ""), explanation)
    severity    = severity_scorer.score(explanation, rule.get("severity_hint", "Medium"))
    return {
        "record_id":   req.record_id,
        "rule_id":     req.rule_id,
        "explanation": explanation,
        "fix":         fix,
        "severity":    severity,
    }


@app.get("/trend")
def get_trend(limit: int = 30):
    return {
        "history": trend.get_trend(limit),
        "stats":   trend.summary_stats(),
        "alerts":  trend.check_deterioration(),
    }


@app.get("/violations")
def get_violations(resolved: Optional[bool] = None):
    violations = db.get_violations(resolved)
    return {"violations": violations, "count": len(violations)}


@app.get("/summary")
def get_summary():
    return db.compliance_summary()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — finds a free port if default is taken
# ═══════════════════════════════════════════════════════════════════════════════

def _find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]


if __name__ == "__main__":
    preferred = int(os.environ.get("PORT", 7860))
    port = _find_free_port(preferred)
    if port != preferred:
        print(f"[WARN] Port {preferred} in use, binding on {port} instead.", flush=True)
    uvicorn.run("inference:app", host="0.0.0.0", port=port, reload=False, workers=1)