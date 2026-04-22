"""
backend/privacy_engine.py

Presidio-based PII anonymization engine.
Uses en_core_web_sm model.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationResult:
    anonymized_text:   str
    detected_entities: list[str]     = field(default_factory=list)
    entity_count:      int           = 0
    pii_found:         bool          = False
    entities_csv:      Optional[str] = None


class PrivacyEngine:

    ENTITIES_TO_DETECT = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
        "DATE_TIME", "IP_ADDRESS", "URL", "NRP",
        "IN_PAN", "IN_AADHAAR", "IN_VOTER", "IN_PASSPORT",
    ]

    INDIAN_NAMES = [
        "aarav","aditya","akash","amit","ananya","anjali","ankit","ansh",
        "arjun","aryan","ayaan","deepak","devansh","dhruv","divya","gaurav",
        "harsh","ishaan","ishita","jatin","kabir","kartik","kavya","kiran",
        "krishna","kunal","lakshmi","manish","meera","mohit","neha","nikhil",
        "nikita","nisha","om","pooja","prateek","priya","rahul","rajesh",
        "rajan","ravi","ritika","rohit","rohan","sakshi","sanjay","sara",
        "sarthak","shivam","shreya","simran","sneha","suresh","tanvi",
        "tushar","uday","varun","vidya","vikram","vishal","yash","zara",
    ]

    def __init__(self):
        import time
        t0 = time.time()

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider       = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine_obj = provider.create_engine()
        self.analyzer  = AnalyzerEngine(nlp_engine=nlp_engine_obj)

        self._add_custom_recognizers()

        self.anonymizer = AnonymizerEngine()
        elapsed = round((time.time() - t0) * 1000)
        logger.info(f"PrivacyEngine ready — model=en_core_web_sm, init={elapsed}ms")

    def _add_custom_recognizers(self) -> None:
        """Add India-specific and name-pattern recognizers."""

        # Aadhaar number
        self.analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="IN_AADHAAR",
            patterns=[
                Pattern("aadhaar_spaced", r"\b[0-9]{4}\s[0-9]{4}\s[0-9]{4}\b", 0.95),
                Pattern("aadhaar_plain",  r"\b[0-9]{12}\b",                     0.75),
                Pattern("aadhaar_hyphen", r"\b[0-9]{4}-[0-9]{4}-[0-9]{4}\b",   0.95),
            ],
        ))

        # PAN card
        self.analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="IN_PAN",
            patterns=[
                Pattern("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.95),
            ],
        ))

        # Common Indian first names (keyword list — compensates for sm model's weaker NER)
        self.analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PERSON",
            deny_list=self.INDIAN_NAMES,
            deny_list_score=0.6,
        ))

        # "my name is X" / "I am X" / "call me X" pattern
        self.analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PERSON",
            patterns=[
                Pattern(
                    "name_intro",
                    r"(?i)(?:my name is|i am|i'm|call me|this is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
                    0.85,
                ),
            ],
        ))

        logger.info("Custom recognizers added (Aadhaar, PAN, Indian names, name-intro pattern)")

    def anonymize(self, text: str, language: str = "en") -> AnonymizationResult:
        if not text or not text.strip():
            return AnonymizationResult(anonymized_text=text)

        try:
            results = self.analyzer.analyze(
                text=text,
                entities=self.ENTITIES_TO_DETECT,
                language=language,
            )

            if not results:
                return AnonymizationResult(anonymized_text=text, pii_found=False)

            entity_types = list({r.entity_type for r in results})
            operators    = {
                et: OperatorConfig("replace", {"new_value": f"<{et}>"})
                for et in self.ENTITIES_TO_DETECT
            }

            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators,
            )

            return AnonymizationResult(
                anonymized_text   = anonymized.text,
                detected_entities = entity_types,
                entity_count      = len(results),
                pii_found         = True,
                entities_csv      = ",".join(entity_types),
            )

        except Exception as e:
            logger.error(f"Anonymization failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Anonymization failed — request aborted to prevent PII leakage: {e}"
            )

    def get_pii_risk_level(self, result: AnonymizationResult) -> str:
        if not result.pii_found:
            return "NONE"
        entities = set(result.detected_entities)
        if entities & {"IN_AADHAAR", "IN_PAN", "IN_VOTER", "IN_PASSPORT"}:
            return "HIGH"
        if entities & {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "IP_ADDRESS"}:
            return "MEDIUM"
        return "LOW"


# Singleton — imported by main.py
privacy_engine = PrivacyEngine()