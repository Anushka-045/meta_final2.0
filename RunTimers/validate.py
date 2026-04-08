"""
validate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pre-submission validation for RunTimers pipeline component.
Run this before pushing to HuggingFace Spaces.

  python validate.py

All checks must pass (✓) before submitting.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

PASS = "  ✓"
FAIL = "  ✗"
results = []

def check(name, fn):
    try:
        ok, msg = fn()
        icon = PASS if ok else FAIL
        print(f"{icon}  {name}: {msg}")
        results.append((name, ok, msg))
    except Exception as e:
        print(f"{FAIL}  {name}: EXCEPTION — {e}")
        results.append((name, False, str(e)))


# ── 1. openenv.yaml ──────────────────────────────────────────────────────────
def check_yaml():
    import yaml
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    required = ["name", "version", "tasks", "reward", "api"]
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"Missing keys: {missing}"
    if len(data["tasks"]) < 3:
        return False, f"Only {len(data['tasks'])} tasks, need 3+"
    return True, f"{len(data['tasks'])} tasks, all keys present"
check("openenv.yaml valid", check_yaml)


# ── 2. Database initialises with correct record counts ───────────────────────
def check_database():
    from database.company_db import CompanyDatabase
    import tempfile, os as _os
    tmp = tempfile.mktemp(suffix=".sqlite")
    try:
        db = CompanyDatabase(tmp)
        records = db.get_all_records()
        employees = [r for r in records if r["type"] == "employee"]
        contracts = [r for r in records if r["type"] == "contract"]
        transactions = [r for r in records if r["type"] == "transaction"]
        rules = db.get_rules()
        assert len(employees) == 30, f"Expected 30 employees, got {len(employees)}"
        assert len(contracts) == 15, f"Expected 15 contracts, got {len(contracts)}"
        assert len(transactions) == 15, f"Expected 15 transactions, got {len(transactions)}"
        assert len(rules) == 10, f"Expected 10 rules, got {len(rules)}"
    finally:
        _os.unlink(tmp)
    return True, f"30 employees, 15 contracts, 15 transactions, 10 rules"
check("Database seed data", check_database)


# ── 3. Scanner detects all 10 ground-truth violations ───────────────────────
def check_scanner():
    from database.company_db import CompanyDatabase
    from pipeline.scanner import ComplianceScanner
    import tempfile, os as _os

    EXPECTED = [
        ("EMP001", "RULE001"), ("EMP001", "RULE002"), ("EMP010", "RULE003"),
        ("EMP005", "RULE004"), ("EMP015", "RULE005"), ("CON003", "RULE006"),
        ("CON008", "RULE007"), ("TXN004", "RULE008"), ("TXN009", "RULE009"),
        ("EMP020", "RULE010"),
    ]

    tmp = tempfile.mktemp(suffix=".sqlite")
    try:
        db = CompanyDatabase(tmp)
        s = ComplianceScanner()
        violations = s.scan(db.get_all_records(), db.get_rules())
        found = {(v["record_id"], v["rule_id"]) for v in violations}
        missed = [e for e in EXPECTED if e not in found]
    finally:
        _os.unlink(tmp)

    if missed:
        return False, f"Missed violations: {missed}"
    return True, f"All 10 ground-truth violations detected ({len(violations)} total)"
check("Scanner detects all 10 violations", check_scanner)


# ── 4. Severity scoring works ─────────────────────────────────────────────────
def check_severity():
    from pipeline.scanner import ComplianceScanner
    from database.company_db import CompanyDatabase
    import tempfile, os as _os

    tmp = tempfile.mktemp(suffix=".sqlite")
    SEV_ORDER = ["Low", "Medium", "High", "Critical"]
    try:
        db = CompanyDatabase(tmp)
        s = ComplianceScanner()
        violations = s.scan(db.get_all_records(), db.get_rules())
        bad = [v for v in violations if v.get("severity") not in SEV_ORDER]
    finally:
        _os.unlink(tmp)

    if bad:
        return False, f"{len(bad)} violations have invalid severity"
    return True, f"All {len(violations)} violations have valid severity labels"
check("Severity scoring", check_severity)


# ── 5. Trend tracker records and detects deterioration ───────────────────────
def check_trend():
    from pipeline.trend_tracker import TrendTracker
    tracker = TrendTracker(db=None)

    # Simulate improving then deteriorating scores
    tracker.record({"violations": [{"severity": "High"}] * 3, "total_records": 60})
    tracker.record({"violations": [{"severity": "Critical"}] * 8, "total_records": 60})

    history = tracker.get_trend()
    assert len(history) >= 2, "Expected at least 2 trend entries"

    alerts = tracker.check_deterioration()
    stats = tracker.summary_stats()
    assert "latest_score" in stats

    return True, f"{len(history)} trend entries, {len(alerts)} alert(s) detected"
check("Trend tracker", check_trend)


# ── 6. Conflict detection works ───────────────────────────────────────────────
def check_conflicts():
    from pipeline.scanner import ComplianceScanner
    from database.default_rules import DEFAULT_COMPLIANCE_RULES, CONFLICTING_RULES

    s = ComplianceScanner()
    all_rules = DEFAULT_COMPLIANCE_RULES + CONFLICTING_RULES
    conflicts = s.detect_policy_conflicts(all_rules)
    known = {"RULE005", "RULE_C1", "RULE008", "RULE_C2"}
    found_ids = {id_ for a, b, _ in conflicts for id_ in (a["id"], b["id"])}
    detected = known & found_ids
    return True, f"{len(conflicts)} conflicts found, known pairs detected: {detected}"
check("Conflict detection", check_conflicts)


# ── 7. app.py exists and imports correctly ───────────────────────────────────
def check_app_py():
    with open("app.py") as f:
        src = f.read()
    required = ["from runtimers_server import app", "uvicorn.run", "7860"]
    missing = [k for k in required if k not in src]
    if missing:
        return False, f"Missing from app.py: {missing}"
    return True, "app.py has all required elements"
check("app.py (HF Spaces entry point)", check_app_py)


# ── 8. Dockerfile valid for HF Spaces ────────────────────────────────────────
def check_dockerfile():
    with open("Dockerfile") as f:
        src = f.read()
    checks = ["FROM python", "EXPOSE 7860", "CMD", "appuser", "7860"]
    missing = [c for c in checks if c not in src]
    if missing:
        return False, f"Dockerfile missing: {missing}"
    return True, "Dockerfile valid (non-root user, port 7860)"
check("Dockerfile (HF Spaces ready)", check_dockerfile)


# ── 9. merge_bridge.py has correct structure ──────────────────────────────────
def check_merge_bridge():
    with open("merge_bridge.py") as f:
        src = f.read()
    required = ["MERGE_AVAILABLE", "load_real_data", "PIPELINE_URL",
                "openenv/records", "openenv/rules"]
    missing = [k for k in required if k not in src]
    if missing:
        return False, f"Missing from merge_bridge.py: {missing}"
    return True, "merge_bridge.py has all required elements"
check("merge_bridge.py (Gargi integration)", check_merge_bridge)


# ── 10. requirements file complete ───────────────────────────────────────────
def check_requirements():
    with open("requirements_runtimers.txt") as f:
        src = f.read()
    required = ["fastapi", "uvicorn", "pydantic", "anthropic", "pdfplumber"]
    missing = [k for k in required if k not in src]
    if missing:
        return False, f"Missing packages: {missing}"
    return True, f"All required packages listed"
check("requirements_runtimers.txt", check_requirements)


# ── 11. Static dashboard exists ───────────────────────────────────────────────
def check_dashboard():
    import os
    path = os.path.join("static", "dashboard.html")
    if not os.path.exists(path):
        return False, "static/dashboard.html not found"
    size = os.path.getsize(path)
    if size < 5000:
        return False, f"dashboard.html too small ({size} bytes)"
    with open(path) as f:
        src = f.read()
    required = ["runScan", "uploadPDF", "compliance-score"]
    missing = [k for k in required if k not in src]
    if missing:
        return False, f"Dashboard missing: {missing}"
    return True, f"dashboard.html valid ({size//1024}KB)"
check("Dashboard UI", check_dashboard)


# ── 12. FastAPI server imports cleanly ────────────────────────────────────────
def check_server_import():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("runtimers_server", "runtimers_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "app"), "FastAPI app not found"
    routes = [r.path for r in mod.app.routes]
    required_routes = ["/health", "/records", "/rules", "/scan", "/openenv/records", "/openenv/rules"]
    missing = [r for r in required_routes if r not in routes]
    if missing:
        return False, f"Missing routes: {missing}"
    return True, f"Server imports OK, {len(routes)} routes registered"
check("FastAPI server", check_server_import)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"VALIDATION RESULT: {passed}/{total} checks passed")

if passed == total:
    print("  ✓  ALL CHECKS PASSED — ready to push to HuggingFace Spaces!\n")
    print("Next step:")
    print("  git add .")
    print("  git commit -m 'feat: RunTimers compliance pipeline complete'")
    print("  git push hf main")
else:
    print("\nFix these before submitting:")
    for name, ok, msg in results:
        if not ok:
            print(f"  ✗  {name}: {msg}")

sys.exit(0 if passed == total else 1)
