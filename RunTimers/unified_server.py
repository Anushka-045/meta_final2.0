"""
unified_server.py — Single FastAPI app for HuggingFace Spaces (port 7860).
Combines the OpenEnv environment endpoints (POST /reset, POST /step, GET /state, GET /tasks)
with the compliance pipeline dashboard and API endpoints.
"""
import os, sys, datetime
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
env = ComplianceEnvironment()

_db_path = os.environ.get("DB_PATH", "compliance_db.sqlite")
db = CompanyDatabase(_db_path)
scanner = ComplianceScanner()
trend = TrendTracker(db=db)
api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
pipeline = CompliancePipeline(api_key=api_key) if (_pipeline_available and api_key) else None
explainer = ViolationExplainer(api_key=api_key) if (_pipeline_available and api_key) else None
severity_scorer = SeverityScorer() if _pipeline_available else None
STATIC_DIR = Path(__file__).parent / "static"


# ═══════════════════════════════════════════════════════════════════════════════
# OpenEnv endpoints  (required by the validator)
# ═══════════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    task_id: Literal["task_easy", "task_medium", "task_hard"] = "task_easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "compliance-monitor",
        "version": "1.0.0",
        "pdf_pipeline": _pipeline_available,
        "llm_available": bool(api_key),
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        result = env.reset(task_id=req.task_id, seed=req.seed or 42)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        result = env.step(req.action)
        return result
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
# Pipeline / dashboard endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class ScanRequest(BaseModel):
    record_type: Optional[str] = None
    include_explanations: bool = False


class ConflictRequest(BaseModel):
    rule_ids: Optional[List[str]] = None


class ViolationExplainRequest(BaseModel):
    record_id: str
    rule_id: str


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    p = STATIC_DIR / "dashboard.html"
    if p.exists():
        return HTMLResponse(content=p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Compliance Monitor API</h1><p>See <a href='/docs'>/docs</a></p>")


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
        raise HTTPException(status_code=503, detail="Set HF_TOKEN (or OPENAI_API_KEY) and install pdfplumber")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    try:
        pdf_bytes = await file.read()
        result = pipeline.ingest_pdf_bytes(pdf_bytes, file.filename)
        for rule in result["rules"]:
            db.insert_rule(rule, source=f"pdf:{file.filename}")
        env.add_rules(result["rules"])  # <-- add PDF rules to the environment
        return {
            "source": file.filename,
            "rules_extracted": result["rules_count"],
            "rules": result["rules"],
            "ingested_at": result["ingested_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/scan")
def run_scan(req: ScanRequest = ScanRequest()):
    records = db.get_all_records(req.record_type)
    rules = db.get_rules()
    raw_violations = scanner.scan(records, rules)
    rule_map = {r["id"]: r for r in rules}
    record_map = {r["id"]: r for r in records}
    violations = []
    for v in raw_violations:
        out = {"record_id": v["record_id"], "record_type": v["record_type"],
               "rule_id": v["rule_id"], "rule_category": v["rule_category"],
               "severity": v["severity"], "detail": v["detail"], "flagged_at": v["flagged_at"]}
        if req.include_explanations and explainer and api_key:
            record = record_map.get(v["record_id"], {})
            rule = rule_map.get(v["rule_id"], {})
            try:
                explanation = explainer.explain(record, rule)
                fix = explainer.suggest_fix(v["record_id"], rule.get("text", ""), explanation)
                out["explanation"] = explanation
                out["fix"] = fix
                out["severity"] = severity_scorer.score(explanation, v["severity"])
            except Exception:
                out["explanation"] = v["detail"]
                out["fix"] = "Review and remediate this record."
        violations.append(out)
        db.log_violation(out)
    scan_result = {"violations": violations, "total_records": len(records),
                   "scanned_at": datetime.datetime.utcnow().isoformat()}
    trend.record(scan_result)
    return {**scan_result, "violation_count": len(violations),
            "alerts": trend.check_deterioration(), "summary": db.compliance_summary()}


@app.post("/scan/conflicts")
def detect_rule_conflicts(req: ConflictRequest = ConflictRequest()):
    rules = db.get_rules()
    if req.rule_ids:
        rules = [r for r in rules if r["id"] in req.rule_ids]
    conflicts_raw = scanner.detect_policy_conflicts(rules + CONFLICTING_RULES)
    return {"conflicts": [{"rule_id_a": a["id"], "rule_id_b": b["id"], "description": desc,
                           "rule_a_text": a.get("text", ""), "rule_b_text": b.get("text", "")}
                          for a, b, desc in conflicts_raw],
            "conflict_count": len(conflicts_raw)}


@app.post("/explain")
def explain_violation(req: ViolationExplainRequest):
    if not explainer or not api_key:
        raise HTTPException(status_code=503, detail="LLM explainer unavailable.")
    record = db.get_record(req.record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {req.record_id} not found")
    rules = db.get_rules()
    rule = next((r for r in rules if r["id"] == req.rule_id), None)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {req.rule_id} not found")
    explanation = explainer.explain(record, rule)
    fix = explainer.suggest_fix(req.record_id, rule.get("text", ""), explanation)
    severity = severity_scorer.score(explanation, rule.get("severity_hint", "Medium"))
    return {"record_id": req.record_id, "rule_id": req.rule_id,
            "explanation": explanation, "fix": fix, "severity": severity}


@app.get("/trend")
def get_trend(limit: int = 30):
    return {"history": trend.get_trend(limit), "stats": trend.summary_stats(),
            "alerts": trend.check_deterioration()}


@app.get("/violations")
def get_violations(resolved: Optional[bool] = None):
    violations = db.get_violations(resolved)
    return {"violations": violations, "count": len(violations)}


@app.get("/summary")
def get_summary():
    return db.compliance_summary()


# ─── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("[INFO] Starting FastAPI server...")

    uvicorn.run(
        "unified_server:app",   # file_name:app_variable
        host="0.0.0.0",
        port=7860,
        reload=False
    )