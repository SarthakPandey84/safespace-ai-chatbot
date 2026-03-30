import logging
from typing import Optional
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AnonymizationResult:
    anonymized_text:   str
    detected_entities: list[str]         = field(default_factory=list)
    entity_count:      int               = 0
    pii_found:         bool              = False
    entities_csv:      Optional[str]     = None


class PrivacyEngine:
    ENTITIES_TO_DETECT = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", 
        "DATE_TIME", "IP_ADDRESS", "URL", "NRP", "MEDICAL_LICENSE", 
        "AGE", "IN_PAN", "IN_AADHAAR", "IN_VOTER", "IN_PASSPORT"
    ]

    def __init__(self):
        import time
        start_time = time.time()

        logger.info("Initializing PrivacyEngine with en_core_web_sm...")

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine_obj = provider.create_engine()
        
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine_obj)

        from presidio_analyzer import PatternRecognizer, Pattern
        aadhaar_recognizer = PatternRecognizer(
            supported_entity = "IN_AADHAAR",
            patterns = [
                Pattern(name="aadhaar_spaced", regex=r"\b[0-9]{4}\s[0-9]{4}\s[0-9]{4}\b", score=0.95),
                Pattern(name="aadhaar_plain", regex=r"\b[0-9]{12}\b", score=0.75),
                Pattern(name="aadhaar_hyphen", regex=r"\b[0-9]{4}-[0-9]{4}-[0-9]{4}\b", score=0.95),
            ]
        )
        self.analyzer.registry.add_recognizer(aadhaar_recognizer)
        self.anonymizer = AnonymizerEngine()

        elapsed = round((time.time() - start_time) * 1000)
        logger.info(f"PrivacyEngine initialized in {elapsed}ms.")

    def anonymize(self, text: str, language: str = "en") -> AnonymizationResult:
        if not text or not text.strip():
            return AnonymizationResult(anonymized_text=text)

        try:
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=self.ENTITIES_TO_DETECT,
                language=language
            )

            if not analyzer_results:
                return AnonymizationResult(anonymized_text=text, pii_found=False)

            detected_entity_types = list({result.entity_type for result in analyzer_results})
            entity_count = len(analyzer_results)

            operators = {
                entity_type: OperatorConfig("replace", {"new_value": f"<{entity_type}>"})
                for entity_type in self.ENTITIES_TO_DETECT
            }

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )

            return AnonymizationResult(
                anonymized_text   = anonymized_result.text,
                detected_entities = detected_entity_types,
                entity_count      = entity_count,
                pii_found         = True,
                entities_csv      = ",".join(detected_entity_types) if detected_entity_types else None
            )

        except Exception as e:
            logger.error(f"Anonymization pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Anonymization failed. Request aborted to prevent PII leakage: {e}")

    def get_pii_risk_level(self, result: AnonymizationResult) -> str:
        if not result.pii_found:
            return "NONE"

        entities = set(result.detected_entities)
        
        if entities & {"IN_AADHAAR", "IN_PAN", "IN_VOTER", "IN_PASSPORT", "MEDICAL_LICENSE"}:
            return "HIGH"
        if entities & {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "IP_ADDRESS"}:
            return "MEDIUM"
            
        return "LOW"


privacy_engine = PrivacyEngine()


if __name__ == "__main__":
    test_cases = [
        "I have been feeling really anxious lately.",
        "Hi, I'm Priya Sharma. You can reach me at priya.s@gmail.com or 9876543210.",
        "My Aadhaar number is 1234 5678 9012."
    ]

    for i, text in enumerate(test_cases, 1):
        result = privacy_engine.anonymize(text)
        print(f"\n[Test {i}] Input: {text}")
        print(f"Output: {result.anonymized_text}")
        print(f"Entities: {result.detected_entities}")