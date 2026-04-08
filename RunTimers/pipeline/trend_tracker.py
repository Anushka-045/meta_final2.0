"""
Trend Tracker
Logs compliance scores over time and flags deterioration.
"""
import datetime
from typing import List, Dict, Optional


class TrendTracker:
    """
    Tracks compliance health over time and emits alerts when scores deteriorate.
    Works on top of CompanyDatabase.compliance_trend table.
    """

    DETERIORATION_THRESHOLD = 0.05   # 5% score drop triggers alert
    CRITICAL_SPIKE_THRESHOLD = 2     # 2+ new Critical violations triggers alert

    def __init__(self, db=None):
        """Pass a CompanyDatabase instance, or None to use in-memory only."""
        self.db = db
        self._memory_log: List[Dict] = []

    def record(self, scan_result: Dict):
        """Log a scan result."""
        entry = self._build_entry(scan_result)
        self._memory_log.append(entry)
        if self.db:
            self.db.record_trend(scan_result)

    def _build_entry(self, scan_result: Dict) -> Dict:
        violations = scan_result.get("violations", [])
        total = scan_result.get("total_records", 0)
        sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for v in violations:
            sev = v.get("severity", "Medium")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
        return {
            "scan_time": datetime.datetime.utcnow().isoformat(),
            "total_records": total,
            "total_violations": len(violations),
            "compliance_score": round(1.0 - len(violations) / max(total, 1), 4),
            **{f"{k.lower()}_count": v for k, v in sev_counts.items()},
        }

    def get_trend(self, limit: int = 30) -> List[Dict]:
        """Return recent trend data (from DB if available, else in-memory)."""
        if self.db:
            return self.db.get_trend(limit)
        return self._memory_log[-limit:]

    def check_deterioration(self) -> List[Dict]:
        """
        Analyse recent trend and return a list of alert dicts if deterioration detected.
        """
        history = self.get_trend(10)
        alerts = []
        if len(history) < 2:
            return alerts

        # Most recent two scans
        latest = history[0]
        previous = history[1]

        score_diff = previous["compliance_score"] - latest["compliance_score"]
        if score_diff >= self.DETERIORATION_THRESHOLD:
            alerts.append({
                "type": "score_deterioration",
                "message": (f"Compliance score dropped {score_diff:.1%} "
                            f"({previous['compliance_score']:.1%} → {latest['compliance_score']:.1%})"),
                "severity": "High" if score_diff >= 0.10 else "Medium",
                "detected_at": datetime.datetime.utcnow().isoformat(),
            })

        critical_delta = latest.get("critical_count", 0) - previous.get("critical_count", 0)
        if critical_delta >= self.CRITICAL_SPIKE_THRESHOLD:
            alerts.append({
                "type": "critical_spike",
                "message": (f"{critical_delta} new Critical violation(s) detected "
                            f"since last scan"),
                "severity": "Critical",
                "detected_at": datetime.datetime.utcnow().isoformat(),
            })

        high_delta = latest.get("high_count", 0) - previous.get("high_count", 0)
        if high_delta >= 3:
            alerts.append({
                "type": "high_violations_spike",
                "message": f"{high_delta} new High severity violations detected",
                "severity": "High",
                "detected_at": datetime.datetime.utcnow().isoformat(),
            })

        return alerts

    def summary_stats(self) -> Dict:
        """Return aggregate statistics across all tracked scans."""
        history = self.get_trend(100)
        if not history:
            return {"message": "No scan history available"}

        scores = [h["compliance_score"] for h in history]
        return {
            "scans_recorded": len(history),
            "latest_score": scores[0] if scores else None,
            "avg_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "trend": "improving" if len(scores) >= 2 and scores[0] > scores[-1]
                     else "deteriorating" if len(scores) >= 2 and scores[0] < scores[-1]
                     else "stable",
        }