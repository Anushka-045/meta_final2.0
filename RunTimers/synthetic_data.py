"""
Synthetic company data generator.
Produces deterministic fake employees, contracts, and transactions.
"""
import json, random, datetime

random.seed(42)

DEPARTMENTS = ["Engineering", "HR", "Finance", "Legal", "Operations", "Sales"]
ROLES = ["Junior Engineer", "Senior Engineer", "Manager", "Director", "Analyst", "Contractor"]
COUNTRIES = ["US", "UK", "DE", "IN", "SG", "FR"]

def _rand_date(start_year=2020, end_year=2024):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, delta))).isoformat()

EMPLOYEES = [
    {"id": f"EMP{i:03d}", "name": f"Employee_{i}", "department": random.choice(DEPARTMENTS),
     "role": random.choice(ROLES), "salary": random.randint(40000, 180000),
     "country": random.choice(COUNTRIES), "age": random.randint(22, 65),
     "background_check": random.choice([True, True, True, False]),
     "nda_signed": random.choice([True, True, False]),
     "training_completed": random.choice([True, True, False]),
     "hire_date": _rand_date(), "contract_type": random.choice(["permanent", "contract", "intern"]),
     "access_level": random.randint(1, 5)}
    for i in range(1, 31)
]

# Inject known violations for deterministic grading
EMPLOYEES[0]["background_check"] = False   # EMP001 - clear violation
EMPLOYEES[0]["nda_signed"] = False
EMPLOYEES[4]["salary"] = 220000            # EMP005 - over pay band
EMPLOYEES[4]["training_completed"] = False
EMPLOYEES[9]["age"] = 16                   # EMP010 - underage worker
EMPLOYEES[14]["access_level"] = 5         # EMP015 - high access, contractor
EMPLOYEES[14]["contract_type"] = "contract"
EMPLOYEES[19]["country"] = "SG"           # EMP020 - GDPR issue
EMPLOYEES[19]["nda_signed"] = False

CONTRACTS = [
    {"id": f"CON{i:03d}", "vendor": f"Vendor_{i}", "value": random.randint(5000, 500000),
     "start_date": _rand_date(2020, 2022), "end_date": _rand_date(2023, 2025),
     "approved_by": f"EMP{random.randint(1,30):03d}",
     "dual_approval": random.choice([True, True, False]),
     "gdpr_clause": random.choice([True, True, False]),
     "insurance_verified": random.choice([True, False]),
     "country": random.choice(COUNTRIES)}
    for i in range(1, 16)
]
CONTRACTS[2]["dual_approval"] = False     # CON003 - missing dual approval
CONTRACTS[2]["value"] = 150000
CONTRACTS[7]["gdpr_clause"] = False       # CON008 - GDPR clause missing
CONTRACTS[7]["country"] = "DE"
CONTRACTS[11]["insurance_verified"] = False  # CON012

TRANSACTIONS = [
    {"id": f"TXN{i:03d}", "amount": random.randint(100, 50000),
     "initiated_by": f"EMP{random.randint(1,30):03d}",
     "approved_by": f"EMP{random.randint(1,30):03d}",
     "category": random.choice(["travel", "software", "hardware", "consulting", "marketing"]),
     "date": _rand_date(), "receipt_attached": random.choice([True, True, False]),
     "over_limit": False}
    for i in range(1, 16)
]
TRANSACTIONS[3]["amount"] = 48000         # TXN004 - over approval threshold
TRANSACTIONS[3]["approved_by"] = TRANSACTIONS[3]["initiated_by"]  # self-approved
TRANSACTIONS[3]["over_limit"] = True
TRANSACTIONS[8]["receipt_attached"] = False  # TXN009
TRANSACTIONS[8]["amount"] = 12000

ALL_RECORDS = (
    [{"type": "employee", **e} for e in EMPLOYEES] +
    [{"type": "contract", **c} for c in CONTRACTS] +
    [{"type": "transaction", **t} for t in TRANSACTIONS]
)

COMPLIANCE_RULES = [
    {"id": "RULE001", "category": "HR", "severity_hint": "Critical",
     "text": "All employees must have a completed background check before being granted system access.",
     "applies_to": "employee", "field": "background_check", "condition": "must_be_true"},

    {"id": "RULE002", "category": "HR", "severity_hint": "High",
     "text": "All employees and contractors must sign a Non-Disclosure Agreement (NDA) within 30 days of joining.",
     "applies_to": "employee", "field": "nda_signed", "condition": "must_be_true"},

    {"id": "RULE003", "category": "HR", "severity_hint": "Critical",
     "text": "No employee under the age of 18 may be employed in any capacity.",
     "applies_to": "employee", "field": "age", "condition": "must_be_gte_18"},

    {"id": "RULE004", "category": "HR", "severity_hint": "Medium",
     "text": "All employees must complete mandatory compliance training within 60 days of hire.",
     "applies_to": "employee", "field": "training_completed", "condition": "must_be_true"},

    {"id": "RULE005", "category": "Access", "severity_hint": "High",
     "text": "Contractors must not be assigned access level 4 or higher without Director approval.",
     "applies_to": "employee", "field": "access_level", "condition": "contractor_max_access_3"},

    {"id": "RULE006", "category": "Finance", "severity_hint": "High",
     "text": "Any contract with a value exceeding $100,000 must have dual approval from two senior managers.",
     "applies_to": "contract", "field": "dual_approval", "condition": "must_be_true_if_value_gt_100k"},

    {"id": "RULE007", "category": "Legal/GDPR", "severity_hint": "Critical",
     "text": "All contracts involving EU-based vendors or data subjects must include a GDPR compliance clause.",
     "applies_to": "contract", "field": "gdpr_clause", "condition": "must_be_true_if_eu_country"},

    {"id": "RULE008", "category": "Finance", "severity_hint": "High",
     "text": "Transactions above $10,000 must not be self-approved. A second approver is required.",
     "applies_to": "transaction", "field": "approved_by", "condition": "no_self_approval_above_10k"},

    {"id": "RULE009", "category": "Finance", "severity_hint": "Medium",
     "text": "All expense transactions above $5,000 must have a receipt or invoice attached.",
     "applies_to": "transaction", "field": "receipt_attached", "condition": "must_be_true_if_amount_gt_5k"},

    {"id": "RULE010", "category": "Legal/GDPR", "severity_hint": "High",
     "text": "Personal data of employees in GDPR-protected regions must not be processed without signed NDA.",
     "applies_to": "employee", "field": "nda_signed", "condition": "must_be_true_if_gdpr_region"},
]

# Conflicting rules for Task 3
CONFLICTING_RULES = [
    {"id": "RULE_C1", "category": "Access", "severity_hint": "Medium",
     "text": "Contractors may be granted temporary access level 5 during project critical phases with manager sign-off.",
     "conflicts_with": "RULE005"},
    {"id": "RULE_C2", "category": "Finance", "severity_hint": "Low",
     "text": "For transactions under $50,000 in the marketing category, single approval is sufficient.",
     "conflicts_with": "RULE008"},
]

GROUND_TRUTH_VIOLATIONS = [
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

if __name__ == "__main__":
    print(f"Employees: {len(EMPLOYEES)}, Contracts: {len(CONTRACTS)}, Transactions: {len(TRANSACTIONS)}")
    print(f"Rules: {len(COMPLIANCE_RULES)}, Conflicts: {len(CONFLICTING_RULES)}")
    print(f"Ground truth violations: {len(GROUND_TRUTH_VIOLATIONS)}")
