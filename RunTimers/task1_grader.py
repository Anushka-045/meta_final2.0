"""
Task 1 Grader — Single Record vs Single Rule. Score: 0.0 or 1.0 (binary)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Any

EXPECTED_RECORD_ID = "EMP001"
EXPECTED_RULE_ID   = "RULE001"

def grade(episode_result: Dict[str, Any]) -> float:
    violations = episode_result.get("violations", [])
    for v in violations:
        if v.get("record_id") == EXPECTED_RECORD_ID and v.get("rule_id") == EXPECTED_RULE_ID:
            return 1.0
    return 0.0

def grade_with_details(episode_result: Dict[str, Any]) -> Dict[str, Any]:
    score = grade(episode_result)
    violations = episode_result.get("violations", [])
    found = any(v.get("record_id") == EXPECTED_RECORD_ID and v.get("rule_id") == EXPECTED_RULE_ID
                for v in violations)
    return {"score": score, "task": "task_easy", "details": {
        "expected_violation": f"{EXPECTED_RECORD_ID} x {EXPECTED_RULE_ID}",
        "found": found, "total_flags_raised": len(violations)}}

if __name__ == "__main__":
    assert grade({"violations": [{"record_id": "EMP001", "rule_id": "RULE001"}]}) == 1.0
    assert grade({"violations": []}) == 0.0
    print("Task 1 grader OK:", grade_with_details({"violations": [{"record_id":"EMP001","rule_id":"RULE001","severity":"Critical"}]}))
