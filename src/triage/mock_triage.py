"""Deterministic rule-based Mock Emergency Triage Engine for offline demo reliability."""

import re
from src.triage.interface import TriageEngine
from src.triage.schemas import TriageResult


class MockTriage(TriageEngine):
    """Deterministic, zero-dependency emergency classification and prioritization engine."""

    # Symptom & Scenario Lexicon
    CRITICAL_KEYWORDS = {
        "unconscious": "Patient unconscious / non-responsive",
        "not breathing": "Respiratory arrest / apnea",
        "collapsed": "Sudden collapse",
        "not responding": "Unresponsive patient",
        "chest pain": "Acute cardiac symptoms",
        "heart attack": "Suspected myocardial infarction",
        "bleeding badly": "Severe uncontrolled hemorrhage",
        "severe bleeding": "Severe uncontrolled hemorrhage",
        "cardiac": "Cardiac distress",
        "choking": "Airway obstruction",
        "head trauma": "Severe head trauma",
        "head injury": "Severe head trauma",
    }

    URGENT_KEYWORDS = {
        "accident": "Vehicle or traffic collision",
        "crash": "Impact collision",
        "broken": "Suspected bone fracture",
        "fracture": "Bone fracture",
        "bleeding": "Active bleeding",
        "burning": "Thermal hazard",
        "fire": "Active fire incident",
        "smoke": "Smoke inhalation risk",
        "breathing": "Respiratory difficulty",
        "dizzy": "Altered mental status / vertigo",
        "asthma": "Acute asthma exacerbation",
        "severe pain": "Acute high-severity pain",
        "trapped": "Victim entrapped",
    }

    CATEGORY_PATTERNS = [
        ("TRAFFIC_ACCIDENT", [r"bike", r"car\b", r"accident", r"motorcycle", r"crash", r"collision", r"pedestrian", r"vehicle"]),
        ("FIRE", [r"fire", r"smoke", r"burn", r"explosion", r"flame"]),
        ("CARDIAC", [r"heart", r"chest pain", r"cardiac", r"pulse", r"angina"]),
        ("BREATHING", [r"breath", r"choking", r"asthma", r"suffocat", r"airway", r"gasp"]),
        ("UNCONSCIOUS_PERSON", [r"unconscious", r"collapsed", r"not responding", r"fainted", r"passed out"]),
        ("INJURY", [r"fall", r"bleeding", r"fracture", r"broken", r"wound", r"cut", r"head injury"]),
        ("MEDICAL", [r"fever", r"diabetic", r"allergic", r"seizure", r"poison", r"sick", r"stomach"]),
    ]


    def triage(self, transcript: str) -> TriageResult:
        text = transcript.lower()
        evidence: list[str] = []

        # 1. Evidence extraction
        for kw, desc in self.CRITICAL_KEYWORDS.items():
            if kw in text:
                evidence.append(desc)

        for kw, desc in self.URGENT_KEYWORDS.items():
            if kw in text and desc not in evidence:
                evidence.append(desc)

        # 2. Priority determination
        has_critical = any(kw in text for kw in self.CRITICAL_KEYWORDS)
        has_urgent = any(kw in text for kw in self.URGENT_KEYWORDS)

        if has_critical:
            priority = 3
            summary = "Immediate life-threatening emergency requiring maximum urgency dispatch."
            confidence = min(0.98, 0.88 + len(evidence) * 0.03)
        elif has_urgent:
            priority = 2
            summary = "Urgent emergency requiring rapid response unit."
            confidence = min(0.95, 0.82 + len(evidence) * 0.04)
        else:
            priority = 1
            summary = "Routine medical transport / non-life-threatening situation."
            confidence = 0.85
            evidence.append("Stable vitals / non-acute condition reported")

        # 3. Emergency category classification
        emergency_type = "MEDICAL"
        for cat, patterns in self.CATEGORY_PATTERNS:
            if any(re.search(p, text) for p in patterns):
                emergency_type = cat
                break

        # 4. Optional location extraction e.g. "at 45, 60" or "(72, 31)"
        location_hint = None
        loc_match = re.search(r"\(?(\d{1,2}(?:\.\d+)?)\s*,\s*(\d{1,2}(?:\.\d+)?)\)?", text)
        if loc_match:
            try:
                location_hint = (float(loc_match.group(1)), float(loc_match.group(2)))
            except ValueError:
                pass

        return TriageResult(
            emergency_type=emergency_type,
            priority=priority,
            confidence=confidence,
            evidence=evidence,
            summary=summary,
            location_hint=location_hint,
        )
