"""
Task 3 Grader — Full DB with Conflicting Policies.
Weighted: detection(0.35) + severity(0.25) + explanation(0.20) + fix(0.10) + conflicts(0.10)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Any, List

FULL_GROUND_TRUTH = [
    {"record_id": "EMP001", "rule_id": "RULE001", "expected_severity": "Critical"},
    {"record_id": "EMP001", "rule_id": "RULE002", "expected_severity": "High"},
    {"record_id": "EMP010", "rule_id": "RULE003", "expected_severity": "Critical"},
    {"record_id": "EMP005", "rule_id": "RULE004", "expected_severity": "Medium"},
    {"record_id": "EMP015", "rule_id": "RULE005", "expected_severity": "High"},
    {"record_id": "CON003", "rule_id": "RULE006", "expected_severity": "High"},
    {"record_id": "CON008", "rule_id": "RULE007", "expected_severity": "Critical"},
    {"record_id": "TXN004", "rule_id": "RULE008", "expected_severity": "High"},
    {"record_id": "TXN009", "rule_id": "RULE009", "expected_severity": "Medium"},
    {"record_id": "EMP020", "rule_id": "RULE010", "expected_severity": "High"},
]
KNOWN_CONFLICTS = [("RULE005","RULE_C1"), ("RULE008","RULE_C2")]
SEV_ORDER = ["Low","Medium","High","Critical"]

def _exp_q(s):
    if not s or len(s)<20: return 0.0
    if len(s)>50 and any(k in s.lower() for k in ["violat","policy","record","rule","require","must"]): return 1.0
    return 0.6 if len(s)>30 else 0.3

def _fix_q(s):
    if not s or len(s)<15: return 0.0
    verbs = ["conduct","obtain","update","require","schedule","escalate","review","ensure","complete","assign"]
    return 1.0 if (any(v in s.lower() for v in verbs) and len(s)>25) else (0.6 if len(s)>20 else 0.3)

def grade(episode_result: Dict[str, Any]) -> float:
    violations = episode_result.get("violations", [])
    conflicts  = episode_result.get("conflicts",  [])
    gt_set = {(g["record_id"],g["rule_id"]):g for g in FULL_GROUND_TRUTH}
    tp = fp = 0; sev_s = exp_s = fix_s = 0.0
    for v in violations:
        key = (v.get("record_id"), v.get("rule_id"))
        if key in gt_set:
            tp += 1; gt = gt_set[key]
            if v.get("severity") == gt["expected_severity"]: sev_s += 1.0
            elif v.get("severity") in SEV_ORDER:
                diff = abs(SEV_ORDER.index(v["severity"]) - SEV_ORDER.index(gt["expected_severity"]))
                sev_s += max(0, 1.0-diff*0.5)
            exp_s += _exp_q(v.get("explanation",""))
            fix_s += _fix_q(v.get("fix",""))
        else: fp += 1
    n = len(FULL_GROUND_TRUTH); safe_tp = max(tp,1)
    conflict_s = sum(0.5 for c in conflicts
        if tuple(sorted([c.get("rule_id_a",""),c.get("rule_id_b","")])) in [tuple(sorted(k)) for k in KNOWN_CONFLICTS]
        and len(c.get("resolution",""))>30)
    total = (tp/n)*0.35 + (sev_s/safe_tp*(tp/n))*0.25 + (exp_s/safe_tp*(tp/n))*0.20 + \
            (fix_s/safe_tp*(tp/n))*0.10 + min(0.10, conflict_s*0.10) - min(0.10, fp*0.02)
    return round(max(0.0, min(1.0, total)), 4)

def grade_with_details(episode_result: Dict[str, Any]) -> Dict[str, Any]:
    score = grade(episode_result)
    violations = episode_result.get("violations", [])
    gt_set = {(g["record_id"],g["rule_id"]) for g in FULL_GROUND_TRUTH}
    tp = sum(1 for v in violations if (v.get("record_id"),v.get("rule_id")) in gt_set)
    return {"score": score, "task": "task_hard", "details": {
        "true_positives": tp, "false_positives": len(violations)-tp,
        "ground_truth_total": len(FULL_GROUND_TRUTH),
        "precision": round(tp/max(len(violations),1),3),
        "recall": round(tp/max(len(FULL_GROUND_TRUTH),1),3)}}

if __name__ == "__main__":
    perfect = {"violations": [{"record_id":g["record_id"],"rule_id":g["rule_id"],"severity":g["expected_severity"],
        "explanation":f"Record {g['record_id']} violates rule {g['rule_id']} which requires compliance with policy.",
        "fix":f"Conduct an immediate review and update record {g['record_id']} to ensure full policy compliance."}
        for g in FULL_GROUND_TRUTH],
        "conflicts":[{"rule_id_a":"RULE005","rule_id_b":"RULE_C1","resolution":"RULE005 takes precedence unless Director override documented in writing."},
                     {"rule_id_a":"RULE008","rule_id_b":"RULE_C2","resolution":"Marketing transactions over $10k still require dual approval per RULE008 baseline."}]}
    score = grade(perfect)
    assert score > 0.8, f"Expected >0.8 got {score}"
    print("Task 3 grader OK:", score, grade_with_details(perfect))
