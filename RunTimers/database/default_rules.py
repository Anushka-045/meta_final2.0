"""
Default compliance rules — mirrors Gargi's COMPLIANCE_RULES exactly
so the plug-in merge is seamless.
"""

DEFAULT_COMPLIANCE_RULES = [
    {
        "id": "RULE001",
        "category": "HR",
        "severity_hint": "Critical",
        "text": "All employees must have a completed background check before being granted system access.",
        "applies_to": "employee",
        "field": "background_check",
        "condition": "must_be_true",
    },
    {
        "id": "RULE002",
        "category": "HR",
        "severity_hint": "High",
        "text": "All employees and contractors must sign a Non-Disclosure Agreement (NDA) within 30 days of joining.",
        "applies_to": "employee",
        "field": "nda_signed",
        "condition": "must_be_true",
    },
    {
        "id": "RULE003",
        "category": "HR",
        "severity_hint": "Critical",
        "text": "No employee under the age of 18 may be employed in any capacity.",
        "applies_to": "employee",
        "field": "age",
        "condition": "must_be_gte_18",
    },
    {
        "id": "RULE004",
        "category": "HR",
        "severity_hint": "Medium",
        "text": "All employees must complete mandatory compliance training within 60 days of hire.",
        "applies_to": "employee",
        "field": "training_completed",
        "condition": "must_be_true",
    },
    {
        "id": "RULE005",
        "category": "Access",
        "severity_hint": "High",
        "text": "Contractors must not be assigned access level 4 or higher without Director approval.",
        "applies_to": "employee",
        "field": "access_level",
        "condition": "contractor_max_access_3",
    },
    {
        "id": "RULE006",
        "category": "Finance",
        "severity_hint": "High",
        "text": "Any contract with a value exceeding $100,000 must have dual approval from two senior managers.",
        "applies_to": "contract",
        "field": "dual_approval",
        "condition": "must_be_true_if_value_gt_100k",
    },
    {
        "id": "RULE007",
        "category": "Legal/GDPR",
        "severity_hint": "Critical",
        "text": "All contracts involving EU-based vendors or data subjects must include a GDPR compliance clause.",
        "applies_to": "contract",
        "field": "gdpr_clause",
        "condition": "must_be_true_if_eu_country",
    },
    {
        "id": "RULE008",
        "category": "Finance",
        "severity_hint": "High",
        "text": "Transactions above $10,000 must not be self-approved. A second approver is required.",
        "applies_to": "transaction",
        "field": "approved_by",
        "condition": "no_self_approval_above_10k",
    },
    {
        "id": "RULE009",
        "category": "Finance",
        "severity_hint": "Medium",
        "text": "All expense transactions above $5,000 must have a receipt or invoice attached.",
        "applies_to": "transaction",
        "field": "receipt_attached",
        "condition": "must_be_true_if_amount_gt_5k",
    },
    {
        "id": "RULE010",
        "category": "Legal/GDPR",
        "severity_hint": "High",
        "text": "Personal data of employees in GDPR-protected regions must not be processed without signed NDA.",
        "applies_to": "employee",
        "field": "nda_signed",
        "condition": "must_be_true_if_gdpr_region",
    },
]

# Conflicting rules (used in task_hard)
CONFLICTING_RULES = [
    {
        "id": "RULE_C1",
        "category": "Access",
        "severity_hint": "Medium",
        "text": "Contractors may be granted temporary access level 5 during project critical phases with manager sign-off.",
        "conflicts_with": "RULE005",
    },
    {
        "id": "RULE_C2",
        "category": "Finance",
        "severity_hint": "Low",
        "text": "For transactions under $50,000 in the marketing category, single approval is sufficient.",
        "conflicts_with": "RULE008",
    },
]