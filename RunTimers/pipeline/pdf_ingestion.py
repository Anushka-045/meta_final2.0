"""
PDF Ingestion Pipeline
Reads uploaded policy PDFs, extracts raw text, converts to structured rules using LLM.
"""
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

# ── Optional PDF deps (graceful fallback) ────────────────────────────────────
try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    pdfplumber = None
    PDF_BACKEND = "none"

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAW TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class PDFExtractor:
    """Extracts raw text from PDF files."""

    def extract(self, pdf_path: str) -> str:
        """Return full text content of a PDF."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if PDF_BACKEND == "pdfplumber":
            return self._extract_pdfplumber(str(path))
        else:
            raise RuntimeError(
                "No PDF backend available. Install pdfplumber: pip install pdfplumber"
            )

    def _extract_pdfplumber(self, pdf_path: str) -> str:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        return "\n\n".join(text_parts)

    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "upload.pdf") -> str:
        """Extract text from raw PDF bytes (e.g., uploaded via FastAPI)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            return self.extract(tmp_path)
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LLM-BASED RULE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

RULE_EXTRACTION_PROMPT = """You are a compliance expert. Given the following text from a company policy document, extract all compliance rules.

For each rule, output a JSON object with these exact fields:
- id: string (e.g. "RULE_PDF_001")
- category: string (e.g. "HR", "Finance", "Legal/GDPR", "Access", "Security")
- severity_hint: string — one of "Low", "Medium", "High", "Critical"
- text: string (the rule stated clearly)
- applies_to: string — one of "employee", "contract", "transaction", "general"
- source_page: string (page reference if available, else "unknown")

Return ONLY a valid JSON array of rule objects. No markdown, no explanation.

Policy text:
{policy_text}
"""

CONFLICT_DETECTION_PROMPT = """You are a legal compliance analyst. Compare the two policy excerpts below and identify any contradictions or conflicts between them.

Policy A:
{policy_a}

Policy B:
{policy_b}

For each conflict found, return a JSON object with:
- rule_a_text: string (the conflicting statement from Policy A)
- rule_b_text: string (the conflicting statement from Policy B)  
- conflict_description: string (plain English explanation of the conflict)
- severity: string — one of "Low", "Medium", "High", "Critical"
- resolution_suggestion: string (how to resolve the conflict)

Return ONLY a valid JSON array. If no conflicts, return [].
"""


class RuleExtractor:
    """Converts raw policy text into structured compliance rules using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("MODEL_NAME", model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not _openai_available:
                raise RuntimeError("openai package not installed. Run: pip install openai")
            self._client = OpenAI(api_key=self.api_key or "dummy", base_url=self.api_base)
        return self._client

    def extract_rules(self, policy_text: str, rule_id_prefix: str = "RULE_PDF") -> List[Dict]:
        """Extract structured rules from policy text."""
        prompt = RULE_EXTRACTION_PROMPT.format(policy_text=policy_text[:8000])  # token safety
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        rules = self._parse_json_response(raw)

        # Ensure unique IDs with prefix
        for i, rule in enumerate(rules, 1):
            if not rule.get("id") or not rule["id"].startswith(rule_id_prefix):
                rule["id"] = f"{rule_id_prefix}_{i:03d}"

        return rules

    def detect_conflicts(self, policy_text_a: str, policy_text_b: str) -> List[Dict]:
        """Compare two policy documents and flag contradictions."""
        prompt = CONFLICT_DETECTION_PROMPT.format(
            policy_a=policy_text_a[:4000],
            policy_b=policy_text_b[:4000]
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> List[Dict]:
        """Safely parse JSON from LLM response."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nRaw: {raw[:500]}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VIOLATION EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

EXPLAIN_PROMPT = """You are a compliance officer writing a brief violation report.

Record: {record}
Violated Rule: {rule}

Write 1-2 sentences in plain English explaining:
1. What specific data in this record violates the rule
2. Why this is a problem (business/legal risk)

Be direct and specific. No bullet points. No preamble.
"""

FIX_PROMPT = """You are a compliance officer. A record violates a policy rule.

Record ID: {record_id}
Violated Rule: {rule_text}
Violation Detail: {explanation}

Provide ONE concise recommended action to fix this violation (1 sentence, start with an action verb).
"""


class ViolationExplainer:
    """Generates plain English explanations and fix suggestions for violations."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("MODEL_NAME", model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not _openai_available:
                raise RuntimeError("openai package not installed.")
            self._client = OpenAI(api_key=self.api_key or "dummy", base_url=self.api_base)
        return self._client

    def explain(self, record: Dict, rule: Dict) -> str:
        """Generate plain English explanation for a violation."""
        prompt = EXPLAIN_PROMPT.format(
            record=json.dumps(record, default=str),
            rule=json.dumps(rule, default=str)
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def suggest_fix(self, record_id: str, rule_text: str, explanation: str) -> str:
        """Generate a one-line fix recommendation."""
        prompt = FIX_PROMPT.format(
            record_id=record_id,
            rule_text=rule_text,
            explanation=explanation
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SEVERITY SCORER
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_KEYWORDS = {
    "Critical": [
        "underage", "minor", "child labor", "illegal", "criminal", "fraud",
        "data breach", "gdpr violation", "personal data", "background check", "access control"
    ],
    "High": [
        "nda", "non-disclosure", "dual approval", "self-approval", "contractor access",
        "unauthorized", "missing approval", "high value", "over limit"
    ],
    "Medium": [
        "training", "overdue", "incomplete", "missing receipt", "documentation",
        "policy gap", "late submission"
    ],
    "Low": [
        "minor issue", "cosmetic", "formatting", "low value", "informational"
    ]
}

SEV_ORDER = ["Low", "Medium", "High", "Critical"]


class SeverityScorer:
    """Labels violations Low / Medium / High / Critical."""

    def score(self, violation_text: str, rule_severity_hint: str = "Medium") -> str:
        """Score severity based on text keywords + rule hint."""
        text_lower = violation_text.lower()

        # Keyword-based scoring
        for severity in ["Critical", "High", "Medium", "Low"]:
            if any(kw in text_lower for kw in SEVERITY_KEYWORDS[severity]):
                return severity

        # Fall back to rule hint
        return rule_severity_hint if rule_severity_hint in SEV_ORDER else "Medium"

    def score_batch(self, violations: List[Dict]) -> List[Dict]:
        """Add/update severity field for a list of violations."""
        for v in violations:
            if not v.get("severity"):
                text = f"{v.get('explanation', '')} {v.get('rule_text', '')}"
                v["severity"] = self.score(text, v.get("rule_severity_hint", "Medium"))
        return violations


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class CompliancePipeline:
    """
    End-to-end pipeline:
      PDF → text → rules → scan database → violations → explain → score → fix
    """

    def __init__(self, api_key: Optional[str] = None):
        self.pdf_extractor = PDFExtractor()
        self.rule_extractor = RuleExtractor(api_key=api_key)
        self.explainer = ViolationExplainer(api_key=api_key)
        self.severity_scorer = SeverityScorer()

    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Ingest a PDF and return extracted text + structured rules."""
        raw_text = self.pdf_extractor.extract(pdf_path)
        rules = self.rule_extractor.extract_rules(raw_text)
        return {
            "source": Path(pdf_path).name,
            "raw_text": raw_text,
            "rules": rules,
            "rules_count": len(rules),
            "ingested_at": datetime.datetime.utcnow().isoformat()
        }

    def ingest_pdf_bytes(self, pdf_bytes: bytes, filename: str = "policy.pdf") -> Dict[str, Any]:
        """Ingest PDF from raw bytes."""
        raw_text = self.pdf_extractor.extract_from_bytes(pdf_bytes, filename)
        rules = self.rule_extractor.extract_rules(raw_text)
        return {
            "source": filename,
            "raw_text": raw_text,
            "rules": rules,
            "rules_count": len(rules),
            "ingested_at": datetime.datetime.utcnow().isoformat()
        }

    def compare_policies(self, pdf_path_a: str, pdf_path_b: str) -> List[Dict]:
        """Compare two policy PDFs and return list of conflicts."""
        text_a = self.pdf_extractor.extract(pdf_path_a)
        text_b = self.pdf_extractor.extract(pdf_path_b)
        return self.rule_extractor.detect_conflicts(text_a, text_b)

    def explain_violation(self, record: Dict, rule: Dict) -> Dict:
        """Generate full violation report with explanation, severity, and fix."""
        explanation = self.explainer.explain(record, rule)
        severity = self.severity_scorer.score(explanation, rule.get("severity_hint", "Medium"))
        fix = self.explainer.suggest_fix(
            record_id=record.get("id", "UNKNOWN"),
            rule_text=rule.get("text", ""),
            explanation=explanation
        )
        return {
            "record_id": record.get("id"),
            "rule_id": rule.get("id"),
            "explanation": explanation,
            "severity": severity,
            "fix": fix,
            "flagged_at": datetime.datetime.utcnow().isoformat()
        }