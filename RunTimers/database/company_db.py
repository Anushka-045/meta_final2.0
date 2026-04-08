"""
Synthetic Company Database
Realistic fake employees, contracts, and transactions stored in SQLite.
Mirrors the structure used in Gargi's environment for plug-in compatibility.
"""
import sqlite3
import json
import random
import datetime
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SEED DATA (matches Gargi's synthetic_data.py schema exactly)
# ─────────────────────────────────────────────────────────────────────────────

DEPARTMENTS = ["Engineering", "HR", "Finance", "Legal", "Operations", "Sales"]
ROLES = ["Junior Engineer", "Senior Engineer", "Manager", "Director", "Analyst", "Contractor"]
COUNTRIES = ["US", "UK", "DE", "IN", "SG", "FR"]
EU_COUNTRIES = {"DE", "FR", "IT", "ES", "NL", "BE", "PL", "SE", "AT", "PT"}

def _rand_date(start_year=2020, end_year=2024) -> str:
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, delta))).isoformat()

def _generate_employees() -> List[Dict]:
    employees = [
        {
            "id": f"EMP{i:03d}",
            "name": f"Employee_{i}",
            "department": random.choice(DEPARTMENTS),
            "role": random.choice(ROLES),
            "salary": random.randint(40000, 180000),
            "country": random.choice(COUNTRIES),
            "age": random.randint(22, 65),
            "background_check": random.choice([True, True, True, False]),
            "nda_signed": random.choice([True, True, False]),
            "training_completed": random.choice([True, True, False]),
            "hire_date": _rand_date(),
            "contract_type": random.choice(["permanent", "contract", "intern"]),
            "access_level": random.randint(1, 5),
        }
        for i in range(1, 31)
    ]
    # Inject known violations (mirrors Gargi's data exactly)
    employees[0]["background_check"] = False   # EMP001
    employees[0]["nda_signed"] = False
    employees[4]["salary"] = 220000             # EMP005
    employees[4]["training_completed"] = False
    employees[9]["age"] = 16                    # EMP010
    employees[14]["access_level"] = 5           # EMP015
    employees[14]["contract_type"] = "contract"
    employees[19]["country"] = "SG"             # EMP020
    employees[19]["nda_signed"] = False
    return employees

def _generate_contracts() -> List[Dict]:
    contracts = [
        {
            "id": f"CON{i:03d}",
            "vendor": f"Vendor_{i}",
            "value": random.randint(5000, 500000),
            "start_date": _rand_date(2020, 2022),
            "end_date": _rand_date(2023, 2025),
            "approved_by": f"EMP{random.randint(1, 30):03d}",
            "dual_approval": random.choice([True, True, False]),
            "gdpr_clause": random.choice([True, True, False]),
            "insurance_verified": random.choice([True, False]),
            "country": random.choice(COUNTRIES),
        }
        for i in range(1, 16)
    ]
    contracts[2]["dual_approval"] = False       # CON003
    contracts[2]["value"] = 150000
    contracts[7]["gdpr_clause"] = False         # CON008
    contracts[7]["country"] = "DE"
    contracts[11]["insurance_verified"] = False # CON012
    return contracts

def _generate_transactions() -> List[Dict]:
    transactions = [
        {
            "id": f"TXN{i:03d}",
            "amount": random.randint(100, 50000),
            "initiated_by": f"EMP{random.randint(1, 30):03d}",
            "approved_by": f"EMP{random.randint(1, 30):03d}",
            "category": random.choice(["travel", "software", "hardware", "consulting", "marketing"]),
            "date": _rand_date(),
            "receipt_attached": random.choice([True, True, False]),
            "over_limit": False,
        }
        for i in range(1, 16)
    ]
    transactions[3]["amount"] = 48000           # TXN004
    transactions[3]["approved_by"] = transactions[3]["initiated_by"]
    transactions[3]["over_limit"] = True
    transactions[8]["receipt_attached"] = False  # TXN009
    transactions[8]["amount"] = 12000
    return transactions


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CompanyDatabase:
    """
    SQLite-backed synthetic company database.
    Exposes records in the same dict format as Gargi's environment.
    """

    DEFAULT_PATH = "compliance_db.sqlite"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DEFAULT_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables and seed data if the database is empty."""
        with self._connect() as conn:
            cur = conn.cursor()

            # Employees table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    department TEXT,
                    role TEXT,
                    salary INTEGER,
                    country TEXT,
                    age INTEGER,
                    background_check INTEGER,
                    nda_signed INTEGER,
                    training_completed INTEGER,
                    hire_date TEXT,
                    contract_type TEXT,
                    access_level INTEGER
                )
            """)

            # Contracts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    id TEXT PRIMARY KEY,
                    vendor TEXT,
                    value INTEGER,
                    start_date TEXT,
                    end_date TEXT,
                    approved_by TEXT,
                    dual_approval INTEGER,
                    gdpr_clause INTEGER,
                    insurance_verified INTEGER,
                    country TEXT
                )
            """)

            # Transactions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    amount INTEGER,
                    initiated_by TEXT,
                    approved_by TEXT,
                    category TEXT,
                    date TEXT,
                    receipt_attached INTEGER,
                    over_limit INTEGER
                )
            """)

            # Compliance rules table (populated from PDF ingestion or defaults)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    id TEXT PRIMARY KEY,
                    category TEXT,
                    severity_hint TEXT,
                    text TEXT,
                    applies_to TEXT,
                    field TEXT,
                    condition TEXT,
                    source TEXT DEFAULT 'builtin'
                )
            """)

            # Violations log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS violations_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT,
                    record_type TEXT,
                    rule_id TEXT,
                    severity TEXT,
                    explanation TEXT,
                    fix TEXT,
                    flagged_at TEXT,
                    resolved INTEGER DEFAULT 0
                )
            """)

            # Compliance trend table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS compliance_trend (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_time TEXT,
                    total_records INTEGER,
                    total_violations INTEGER,
                    compliance_score REAL,
                    critical_count INTEGER,
                    high_count INTEGER,
                    medium_count INTEGER,
                    low_count INTEGER
                )
            """)

            conn.commit()

            # Seed data if empty
            if cur.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
                self._seed_data(conn)

    def _seed_data(self, conn: sqlite3.Connection):
        cur = conn.cursor()

        employees = _generate_employees()
        for e in employees:
            cur.execute("""
                INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (e["id"], e["name"], e["department"], e["role"], e["salary"],
                  e["country"], e["age"], int(e["background_check"]),
                  int(e["nda_signed"]), int(e["training_completed"]),
                  e["hire_date"], e["contract_type"], e["access_level"]))

        contracts = _generate_contracts()
        for c in contracts:
            cur.execute("""
                INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (c["id"], c["vendor"], c["value"], c["start_date"], c["end_date"],
                  c["approved_by"], int(c["dual_approval"]),
                  int(c["gdpr_clause"]), int(c["insurance_verified"]), c["country"]))

        transactions = _generate_transactions()
        for t in transactions:
            cur.execute("""
                INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?)
            """, (t["id"], t["amount"], t["initiated_by"], t["approved_by"],
                  t["category"], t["date"], int(t["receipt_attached"]), int(t["over_limit"])))

        # Seed default compliance rules
        from database.default_rules import DEFAULT_COMPLIANCE_RULES
        for r in DEFAULT_COMPLIANCE_RULES:
            cur.execute("""
                INSERT OR IGNORE INTO compliance_rules VALUES (?,?,?,?,?,?,?,?)
            """, (r["id"], r["category"], r["severity_hint"], r["text"],
                  r["applies_to"], r.get("field", ""), r.get("condition", ""), "builtin"))

        conn.commit()

    # ── Query methods ──────────────────────────────────────────────────────

    def get_all_records(self, record_type: Optional[str] = None) -> List[Dict]:
        """Return all records, optionally filtered by type."""
        with self._connect() as conn:
            records = []
            if record_type in (None, "employee"):
                rows = conn.execute("SELECT * FROM employees").fetchall()
                records += [{"type": "employee", **dict(r),
                             "background_check": bool(r["background_check"]),
                             "nda_signed": bool(r["nda_signed"]),
                             "training_completed": bool(r["training_completed"])}
                            for r in rows]
            if record_type in (None, "contract"):
                rows = conn.execute("SELECT * FROM contracts").fetchall()
                records += [{"type": "contract", **dict(r),
                             "dual_approval": bool(r["dual_approval"]),
                             "gdpr_clause": bool(r["gdpr_clause"]),
                             "insurance_verified": bool(r["insurance_verified"])}
                            for r in rows]
            if record_type in (None, "transaction"):
                rows = conn.execute("SELECT * FROM transactions").fetchall()
                records += [{"type": "transaction", **dict(r),
                             "receipt_attached": bool(r["receipt_attached"]),
                             "over_limit": bool(r["over_limit"])}
                            for r in rows]
            return records

    def get_record(self, record_id: str) -> Optional[Dict]:
        """Fetch a single record by ID."""
        prefix = record_id[:3].upper()
        table_map = {"EMP": ("employees", "employee"),
                     "CON": ("contracts", "contract"),
                     "TXN": ("transactions", "transaction")}
        if prefix not in table_map:
            return None
        table, rtype = table_map[prefix]
        with self._connect() as conn:
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (record_id,)).fetchone()
            if row:
                return {"type": rtype, **dict(row)}
        return None

    def get_rules(self, source: Optional[str] = None) -> List[Dict]:
        """Return compliance rules, optionally filtered by source."""
        with self._connect() as conn:
            if source:
                rows = conn.execute(
                    "SELECT * FROM compliance_rules WHERE source = ?", (source,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM compliance_rules").fetchall()
            return [dict(r) for r in rows]

    def insert_rule(self, rule: Dict, source: str = "pdf"):
        """Insert a rule extracted from a PDF."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO compliance_rules VALUES (?,?,?,?,?,?,?,?)
            """, (rule["id"], rule.get("category", "General"),
                  rule.get("severity_hint", "Medium"), rule.get("text", ""),
                  rule.get("applies_to", "general"), rule.get("field", ""),
                  rule.get("condition", ""), source))
            conn.commit()

    def log_violation(self, violation: Dict):
        """Persist a detected violation."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO violations_log
                (record_id, record_type, rule_id, severity, explanation, fix, flagged_at)
                VALUES (?,?,?,?,?,?,?)
            """, (violation.get("record_id"), violation.get("record_type", ""),
                  violation.get("rule_id"), violation.get("severity", "Medium"),
                  violation.get("explanation", ""), violation.get("fix", ""),
                  violation.get("flagged_at", datetime.datetime.utcnow().isoformat())))
            conn.commit()

    def get_violations(self, resolved: Optional[bool] = None) -> List[Dict]:
        """Return logged violations."""
        with self._connect() as conn:
            if resolved is None:
                rows = conn.execute("SELECT * FROM violations_log ORDER BY flagged_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM violations_log WHERE resolved = ? ORDER BY flagged_at DESC",
                    (int(resolved),)
                ).fetchall()
            return [dict(r) for r in rows]

    def record_trend(self, scan_result: Dict):
        """Log a compliance scan result for trend tracking."""
        violations = scan_result.get("violations", [])
        total = scan_result.get("total_records", 0)
        sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for v in violations:
            sev = v.get("severity", "Medium")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        score = round(1.0 - len(violations) / max(total, 1), 4)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO compliance_trend
                (scan_time, total_records, total_violations, compliance_score,
                 critical_count, high_count, medium_count, low_count)
                VALUES (?,?,?,?,?,?,?,?)
            """, (datetime.datetime.utcnow().isoformat(), total, len(violations),
                  score, sev_counts["Critical"], sev_counts["High"],
                  sev_counts["Medium"], sev_counts["Low"]))
            conn.commit()

    def get_trend(self, limit: int = 30) -> List[Dict]:
        """Return recent compliance trend data."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM compliance_trend ORDER BY scan_time DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def compliance_summary(self) -> Dict:
        """High-level compliance health summary."""
        records = self.get_all_records()
        violations = self.get_violations(resolved=False)
        sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for v in violations:
            sev = v.get("severity", "Medium")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        return {
            "total_records": len(records),
            "active_violations": len(violations),
            "compliance_score": round(1.0 - len(violations) / max(len(records), 1), 4),
            "severity_breakdown": sev_counts,
            "rules_loaded": len(self.get_rules()),
            "last_updated": datetime.datetime.utcnow().isoformat()
        }