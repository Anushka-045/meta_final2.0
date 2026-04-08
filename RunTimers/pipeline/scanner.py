"""
Compliance Scanner
Applies rules to all database records and returns structured violations.
This is the core detection engine — no LLM needed for rule evaluation.
"""
import datetime
from typing import List, Dict, Any, Optional, Tuple

EU_COUNTRIES = {"DE", "FR", "IT", "ES", "NL", "BE", "PL", "SE", "AT", "PT", "DK", "FI", "IE"}


class ComplianceScanner:
    """
    Rule-based scanner that checks every record against every applicable rule.
    Returns violations in the same format as Gargi's environment expects.
    """

    def scan(
        self,
        records: List[Dict],
        rules: List[Dict],
    ) -> List[Dict]:
        """
        Scan all records against all rules.
        Returns list of raw violations (without explanations — those come from ViolationExplainer).
        """
        violations = []
        for record in records:
            for rule in rules:
                if not self._rule_applies(record, rule):
                    continue
                violation = self._check_rule(record, rule)
                if violation:
                    violations.append(violation)
        return violations

    def _rule_applies(self, record: Dict, rule: Dict) -> bool:
        """Check if a rule is applicable to a given record type."""
        applies_to = rule.get("applies_to", "general")
        if applies_to == "general":
            return True
        record_type = record.get("type", "")
        return record_type == applies_to

    def _check_rule(self, record: Dict, rule: Dict) -> Optional[Dict]:
        """
        Evaluate a single rule against a record.
        Returns a violation dict if violated, else None.
        """
        condition = rule.get("condition", "")
        record_id = record.get("id", "UNKNOWN")
        rule_id = rule.get("id", "UNKNOWN")

        violated = False
        detail = ""

        # ── HR Rules ────────────────────────────────────────────────────────
        if condition == "must_be_true":
            field = rule.get("field", "")
            val = record.get(field)
            if val is False or val == 0:
                violated = True
                detail = f"Field '{field}' is False/missing for {record_id}"

        elif condition == "must_be_gte_18":
            age = record.get("age", 99)
            if age < 18:
                violated = True
                detail = f"{record_id} has age={age}, below legal minimum of 18"

        elif condition == "contractor_max_access_3":
            ctype = record.get("contract_type", "")
            access = record.get("access_level", 0)
            if ctype == "contract" and access >= 4:
                violated = True
                detail = (f"{record_id} is a contractor (contract_type='{ctype}') "
                          f"with access_level={access} — max allowed is 3")

        # ── Finance / Contract Rules ─────────────────────────────────────────
        elif condition == "must_be_true_if_value_gt_100k":
            value = record.get("value", 0)
            dual = record.get("dual_approval", True)
            if value > 100_000 and not dual:
                violated = True
                detail = (f"{record_id} has value=${value:,} (>$100k) "
                          f"but dual_approval={dual}")

        elif condition == "must_be_true_if_eu_country":
            country = record.get("country", "")
            gdpr = record.get("gdpr_clause", True)
            if country in EU_COUNTRIES and not gdpr:
                violated = True
                detail = (f"{record_id} involves EU country '{country}' "
                          f"but gdpr_clause={gdpr}")

        elif condition == "no_self_approval_above_10k":
            amount = record.get("amount", 0)
            initiated = record.get("initiated_by", "")
            approved = record.get("approved_by", "")
            if amount > 10_000 and initiated == approved:
                violated = True
                detail = (f"{record_id} amount=${amount:,} (>$10k) "
                          f"was self-approved by {initiated}")

        elif condition == "must_be_true_if_amount_gt_5k":
            amount = record.get("amount", 0)
            receipt = record.get("receipt_attached", True)
            if amount > 5_000 and not receipt:
                violated = True
                detail = (f"{record_id} amount=${amount:,} (>$5k) "
                          f"but receipt_attached={receipt}")

        elif condition == "must_be_true_if_gdpr_region":
            country = record.get("country", "")
            nda = record.get("nda_signed", True)
            # SG is in PDPA scope — treat as GDPR-equivalent here per RULE010
            gdpr_scope = EU_COUNTRIES | {"SG", "CA", "BR", "JP", "KR"}
            if country in gdpr_scope and not nda:
                violated = True
                detail = (f"{record_id} in GDPR-scope country '{country}' "
                          f"but nda_signed={nda}")

        if violated:
            return {
                "record_id": record_id,
                "record_type": record.get("type", ""),
                "rule_id": rule_id,
                "rule_category": rule.get("category", ""),
                "severity": rule.get("severity_hint", "Medium"),
                "detail": detail,
                "flagged_at": datetime.datetime.utcnow().isoformat(),
            }
        return None

    def scan_single(self, record: Dict, rules: List[Dict]) -> List[Dict]:
        """Scan a single record against all applicable rules."""
        return [v for v in
                (self._check_rule(record, r) for r in rules if self._rule_applies(record, r))
                if v is not None]

    def detect_policy_conflicts(self, rules: List[Dict]) -> List[Tuple[Dict, Dict, str]]:
        """
        Detect structural conflicts between rules in the same ruleset.
        Returns list of (rule_a, rule_b, conflict_description) tuples.
        """
        conflicts = []
        for i, rule_a in enumerate(rules):
            for rule_b in rules[i + 1:]:
                conflict = self._check_conflict(rule_a, rule_b)
                if conflict:
                    conflicts.append((rule_a, rule_b, conflict))
        return conflicts

    def _check_conflict(self, rule_a: Dict, rule_b: Dict) -> Optional[str]:
        """Heuristic conflict detection between two rules."""
        # Same field, same applies_to, but different conditions
        if (rule_a.get("applies_to") == rule_b.get("applies_to") and
                rule_a.get("field") == rule_b.get("field") and
                rule_a.get("condition") != rule_b.get("condition") and
                rule_a.get("field")):
            return (f"Both rules apply to '{rule_a['applies_to']}.{rule_a['field']}' "
                    f"but with different conditions: '{rule_a['condition']}' vs '{rule_b['condition']}'")

        # Explicit conflict annotation
        if rule_a.get("conflicts_with") == rule_b.get("id"):
            return f"{rule_a['id']} explicitly conflicts with {rule_b['id']}"
        if rule_b.get("conflicts_with") == rule_a.get("id"):
            return f"{rule_b['id']} explicitly conflicts with {rule_a['id']}"

        return None