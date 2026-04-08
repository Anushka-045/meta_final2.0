"""
Task 2 Grader — Multi-Record Multi-Rule. Partial credit scoring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Any, List

MEDIUM_GROUND_TRUTH = [
    {"record_id": "EMP001", "rule_id": "RULE001", "expected_severity": "Critical"},
    {"record_id": "EMP001", "rule_id": "RULE002", "expected_severity": "High"},
    {"record_id": "EMP010", "rule_id": "RULE003", "expected_severity": "Critical"},
    {"record_id": "EMP005", "rule_id": "RULE004", "expected_severity": "Medium"},
]

def grade(episode_result: Dict[str, Any]) -> float:
    violations = episode_result.get("violations", [])
    gt_set = {(g["record_id"], g["rule_id"]): g for g in MEDIUM_GROUND_TRUTH}
    tp = correct_sev = fp = 0
    for v in violations:
        key = (v.get("record_id"), v.get("rule_id"))
        if key in gt_set:
            tp += 1
            if v.get("severity") == gt_set[key]["expected_severity"]:
                correct_sev += 1
        else:
            fp += 1
    n = len(MEDIUM_GROUND_TRUTH)
    raw = tp * 0.4 + correct_sev * 0.2 - fp * 0.1
    return round(max(0.0, min(1.0, raw / (n * 0.6))), 4)

def grade_with_details(episode_result: Dict[str, Any]) -> Dict[str, Any]:
    score = grade(episode_result)
    violations = episode_result.get("violations", [])
    gt_set = {(g["record_id"], g["rule_id"]) for g in MEDIUM_GROUND_TRUTH}
    tp = sum(1 for v in violations if (v.get("record_id"), v.get("rule_id")) in gt_set)
    return {"score": score, "task": "task_medium", "details": {
        "true_positives": tp, "false_positives": len(violations)-tp,
        "ground_truth_total": len(MEDIUM_GROUND_TRUTH),
        "recall": round(tp / max(len(MEDIUM_GROUND_TRUTH),1), 3)}}

if __name__ == "__main__":
    perfect = {"violations": [{"record_id":g["record_id"],"rule_id":g["rule_id"],"severity":g["expected_severity"]} for g in MEDIUM_GROUND_TRUTH]}
    assert grade(perfect) == 1.0, f"Expected 1.0 got {grade(perfect)}"
    assert grade({"violations": []}) == 0.0
    print("Task 2 grader OK:", grade_with_details(perfect))
