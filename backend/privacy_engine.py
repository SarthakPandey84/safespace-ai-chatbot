# ==============================================================================
# FILE: backend/privacy_engine.py
# PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
# PURPOSE: Implements the PII (Personally Identifiable Information) detection
#          and anonymization pipeline using Microsoft Presidio.
#
# ARCHITECTURAL POSITION IN THE PIPELINE:
#   This module sits at Position 2 in the data flow, AFTER the API receives
#   user input and BEFORE any data is sent to the Gemini AI API or logged
#   to the database. This ordering is the physical enforcement of our
#   Privacy-by-Design mandate.
#
#   [User Input] → [FastAPI] → [THIS FILE] → [AI Engine] → [Database]
#                                   ▲
#                            PII is destroyed here.
#                            It never travels further right.
#
# WHAT IS PRIVACY-BY-DESIGN? (Viva Defense Point)
#   Privacy-by-Design (PbD) is a framework coined by Dr. Ann Cavoukian.
#   Its core principle is that privacy protections must be BUILT INTO the
#   system architecture proactively, not added as an afterthought or patch.
#   This file is the concrete implementation of PbD Principle #2:
#   "Privacy as the Default Setting" — the system anonymizes data
#   automatically, and a developer would have to ACTIVELY BREAK this
#   pipeline to send PII to external services.
#
# WHY MICROSOFT PRESIDIO OVER SIMPLE REGEX?
#   A naive approach would use regular expressions (regex) to find patterns
#   like phone numbers or emails. Presidio's advantage is its NLP/NER backbone:
#   it understands CONTEXT. It can identify "John" as a PERSON entity in the
#   sentence "My name is John" — something a regex cannot do. It combines:
#     1. Named Entity Recognition (NER) via SpaCy — for context-aware detection
#     2. Pattern matching (regex) — for structured data like phone/email/SSN
#     3. Rule-based logic — for edge cases and custom entity types
#   This multi-layer approach yields far higher recall (fewer missed PII items)
#   and precision (fewer false positives) than regex alone.
# ==============================================================================

import logging
from typing import Optional
from dataclasses import dataclass, field

# --- Presidio Core Imports ---
# AnalyzerEngine: The NLP-based PII DETECTION engine.
#   Input:  Raw text string
#   Output: A list of RecognizerResult objects, each describing a detected
#           PII span (entity type, start/end character positions, confidence score)
from presidio_analyzer import AnalyzerEngine

# BatchAnalyzerEngine: Allows analyzing multiple texts at once (not used in
# main flow but imported for potential batch research processing use cases).
# AnalyzerEngine is our primary tool.

# AnonymizerEngine: The PII REPLACEMENT engine.
#   Input:  Raw text + list of RecognizerResult objects from the Analyzer
#   Output: Anonymized text with PII replaced by placeholder tags
from presidio_anonymizer import AnonymizerEngine

# AnonymizationConfig & ReplaceConfig: Allow us to customize HOW detected PII
# is replaced. We use ReplaceConfig to wrap entity types in angle brackets,
# e.g., detected PERSON → <PERSON> (readable and informative for researchers).
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: RESULT DATA STRUCTURE
# ==============================================================================

@dataclass
class AnonymizationResult:
    """
    A structured data container (dataclass) for the output of the anonymization
    pipeline. Using a dataclass instead of a plain dictionary provides:
      - Type safety: Fields are typed, catching bugs at development time.
      - Readability: Named fields are self-documenting.
      - IDE support: Autocompletion works correctly with typed fields.

    ACADEMIC NOTE — Dataclasses vs. Dicts vs. Pydantic Models:
        - Plain dict: {'anonymized_text': '...', 'entities': [...]}
          → Flexible but no type safety, easy to misspell keys.
        - dataclass: Lightweight, typed, but no runtime validation.
        - Pydantic BaseModel: Full runtime validation, best for API boundaries.
        We use a dataclass here (internal module boundary) and Pydantic models
        at the API boundary (models.py) — using the right tool for each layer.

    Attributes:
        anonymized_text:   The processed text with PII replaced by tags.
        detected_entities: List of PII entity type strings found (e.g., ['PERSON']).
        entity_count:      Total number of PII instances detected.
        pii_found:         Boolean flag — True if ANY PII was detected.
        entities_csv:      Comma-separated entity types for database storage.
    """
    anonymized_text:   str
    detected_entities: list[str]         = field(default_factory=list)
    entity_count:      int               = 0
    pii_found:         bool              = False
    entities_csv:      Optional[str]     = None


# ==============================================================================
# SECTION 2: THE PRIVACY ENGINE CLASS
# ==============================================================================

class PrivacyEngine:
    """
    Wraps Microsoft Presidio's Analyzer and Anonymizer into a single,
    clean interface with project-specific configuration.

    DESIGN PATTERN — Façade Pattern:
        This class is a Façade. It hides the complexity of initializing two
        separate Presidio engines, configuring entity types, and chaining
        the analyze → anonymize steps. The rest of the application calls
        one simple method: `engine.anonymize(text)`. This separation of
        concerns is what makes the codebase maintainable and testable.

    SINGLETON-LIKE INITIALIZATION:
        Presidio's AnalyzerEngine loads SpaCy's NLP model (en_core_web_lg,
        ~750MB) into memory during __init__. This is an expensive one-time
        cost (3-5 seconds). By instantiating PrivacyEngine ONCE at module
        level and reusing it across all requests, we pay this cost only at
        startup. Instantiating it per-request would make every chat message
        take 3-5 seconds — unacceptable UX.
    """

    # The PII entity types we actively detect and anonymize.
    # This list is derived from Presidio's supported entities for English text.
    # ACADEMIC NOTE: Choosing WHICH entities to detect is an ethical decision,
    # not just a technical one. We err on the side of over-detection (high recall)
    # rather than under-detection, because a missed PII item is a privacy breach,
    # whereas a falsely anonymized word is merely a minor UX inconvenience.
    ENTITIES_TO_DETECT = [
        "PERSON",           # Names (e.g., "John Smith")
        "EMAIL_ADDRESS",    # Emails (e.g., "john@example.com")
        "PHONE_NUMBER",     # Phone numbers (e.g., "+91 9876543210")
        "LOCATION",         # Places (e.g., "Mumbai", "Baker Street")
        "DATE_TIME",        # Dates (e.g., "born on 15th March" could re-identify)
        "IP_ADDRESS",       # IP addresses
        "URL",              # Web URLs that might contain identifying info
        "NRP",              # Nationalities, Religious/Political groups
        "MEDICAL_LICENSE",  # Medical identifiers
        "AGE",              # Age (can be re-identifying in combination with other data)
        "IN_PAN",           # India-specific: Permanent Account Number (PAN card)
        "IN_AADHAAR",       # India-specific: Aadhaar number (national ID)
        "IN_VOTER",         # India-specific: Voter ID
        "IN_PASSPORT",      # India-specific: Passport number
    ]
    # LOCALIZATION NOTE: The IN_* entities are Presidio's built-in recognizers
    # for Indian government ID formats. Since this is an Indian academic project
    # (B.Tech), including these demonstrates thoughtful localization of the
    # privacy framework to the deployment context — a strong viva point.

    def __init__(self):
        """
        Initializes both Presidio engines. This is called ONCE at module load.
        Logs startup time so we can monitor the initialization cost.
        """
        import time
        start_time = time.time()

        logger.info("Initializing PrivacyEngine — loading Presidio + SpaCy NLP model...")
        logger.info("NOTE: First load takes 3-5 seconds due to SpaCy model (en_core_web_lg).")

        # ARCHITECTURE NOTE: AnalyzerEngine auto-discovers all installed
        # Presidio recognizers (including the SpaCy NER-based recognizer and
        # all pattern-based recognizers for structured data like phone/email).
        # Passing no arguments uses the default configuration, which is
        # appropriate for our use case.
        self.analyzer  = AnalyzerEngine()

        # Custom Aadhaar pattern recognizer — added because Presidio's built-in
        # IN_AADHAAR recognizer misses many common formats. This covers both
        # spaced (1234 5678 9012) and plain (123456789012) 12-digit formats.
        from presidio_analyzer import PatternRecognizer, Pattern
        aadhaar_recognizer = PatternRecognizer(
            supported_entity = "IN_AADHAAR",
            patterns = [
                Pattern(
                    name  = "aadhaar_spaced",
                    regex = r"\b[0-9]{4}\s[0-9]{4}\s[0-9]{4}\b",
                    score = 0.95
                ),
                Pattern(
                    name  = "aadhaar_plain",
                    regex = r"\b[0-9]{12}\b",
                    score = 0.75
                ),
                Pattern(
                    name  = "aadhaar_hyphen",
                    regex = r"\b[0-9]{4}-[0-9]{4}-[0-9]{4}\b",
                    score = 0.95
                ),
            ]
        )
        self.analyzer.registry.add_recognizer(aadhaar_recognizer)

        # AnonymizerEngine is stateless and lightweight — it just applies
        # string replacement rules based on the Analyzer's output.
        self.anonymizer = AnonymizerEngine()

        elapsed = round((time.time() - start_time) * 1000)
        logger.info(f"PrivacyEngine initialized successfully in {elapsed}ms.")

    def anonymize(self, text: str, language: str = "en") -> AnonymizationResult:
        """
        The main public method. Takes raw user text, detects all PII,
        replaces it with readable placeholder tags, and returns a structured
        result object.

        TWO-PHASE PIPELINE:
            Phase 1 — ANALYZE:   Find PII → get list of (entity_type, start, end, score)
            Phase 2 — ANONYMIZE: Replace each detected span with <ENTITY_TYPE> tag

        Args:
            text (str):     The raw user input to be anonymized.
            language (str): Language code for NLP model. Defaults to "en" (English).
                            Presidio supports multiple languages; "en" uses SpaCy's
                            English model we installed.

        Returns:
            AnonymizationResult: A structured object containing the anonymized
                                 text and metadata about what was detected.

        Example:
            Input:  "Hi, I'm Priya Sharma, reach me at priya@gmail.com or 9876543210"
            Output: AnonymizationResult(
                        anonymized_text   = "Hi, I'm <PERSON>, reach me at <EMAIL_ADDRESS> or <PHONE_NUMBER>",
                        detected_entities = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'],
                        entity_count      = 3,
                        pii_found         = True,
                        entities_csv      = "PERSON,EMAIL_ADDRESS,PHONE_NUMBER"
                    )
        """
        # --- GUARD CLAUSE: Handle empty or whitespace-only input ---
        # Returning early for trivial input avoids unnecessary NLP processing
        # and prevents potential errors in downstream Presidio calls.
        if not text or not text.strip():
            logger.warning("anonymize() called with empty text. Returning as-is.")
            return AnonymizationResult(anonymized_text=text)

        try:
            # ==================================================================
            # PHASE 1: ANALYZE — Detect PII Entities
            # ==================================================================
            # analyzer_results is a list of RecognizerResult objects, e.g.:
            # [
            #   RecognizerResult(entity_type='PERSON', start=8, end=19, score=0.85),
            #   RecognizerResult(entity_type='EMAIL_ADDRESS', start=33, end=49, score=1.0),
            # ]
            # Each result tells us: what kind of PII, where it is (char positions),
            # and how confident Presidio is (0.0 to 1.0).
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=self.ENTITIES_TO_DETECT,
                language=language
            )

            # If no PII is detected, return the original text unchanged.
            # This is the happy path — most messages won't contain PII.
            if not analyzer_results:
                logger.debug("No PII detected in input text.")
                return AnonymizationResult(
                    anonymized_text=text,
                    pii_found=False
                )

            # Collect the unique entity types detected (for database logging).
            # We use a set first to deduplicate (e.g., two PERSON mentions → one 'PERSON' label).
            detected_entity_types = list({result.entity_type for result in analyzer_results})
            entity_count          = len(analyzer_results)  # Total instances, including duplicates

            logger.info(
                f"PII detected — {entity_count} instance(s) of: {detected_entity_types}. "
                f"Proceeding to anonymization."
            )

            # ==================================================================
            # PHASE 2: ANONYMIZE — Replace PII Spans with Placeholder Tags
            # ==================================================================
            # We configure a CUSTOM operator for each entity type we want to handle.
            # The "replace" operator substitutes the detected span with a custom string.
            #
            # DESIGN CHOICE — <ENTITY_TYPE> format:
            #   We wrap tags in angle brackets (e.g., <PERSON> not PERSON or [PERSON]).
            #   This format is:
            #     1. Visually distinct from normal text — researchers can easily spot redactions.
            #     2. Parseable — scripts can extract entity types with simple regex if needed.
            #     3. Consistent — the same entity always maps to the same placeholder,
            #        enabling frequency analysis (e.g., "how often do users mention locations?")
            #        WITHOUT revealing any actual location names.
            operators = {
                entity_type: OperatorConfig(
                    "replace",
                    {"new_value": f"<{entity_type}>"}
                )
                for entity_type in self.ENTITIES_TO_DETECT
            }

            # The AnonymizerEngine.anonymize() call performs the actual text substitution.
            # It processes ALL detected spans in one pass, handling overlapping entities
            # gracefully (a challenge for naive regex-based approaches).
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )

            # Build and return the structured result.
            entities_csv = ",".join(detected_entity_types) if detected_entity_types else None

            return AnonymizationResult(
                anonymized_text   = anonymized_result.text,
                detected_entities = detected_entity_types,
                entity_count      = entity_count,
                pii_found         = True,
                entities_csv      = entities_csv
            )

        except Exception as e:
            # CRITICAL FAIL-SAFE BEHAVIOR:
            # If the anonymization pipeline throws an unexpected error, we do NOT
            # fall back to sending the raw (unanonymized) text to the AI or database.
            # Instead, we raise the exception, which will cause main.py to return
            # a 500 error to the user.
            #
            # ACADEMIC NOTE — Fail-Safe Defaults (Security Principle):
            #   "Fail-safe defaults" is a foundational security principle stating that
            #   a system should default to a SECURE state on failure, not an insecure one.
            #   A fire door that fails OPEN (letting fire through) is worse than one that
            #   fails CLOSED. Similarly, sending raw PII on pipeline failure is our
            #   "fire door failing open" — an unacceptable privacy breach.
            #   We fail CLOSED: error out rather than leak data.
            logger.error(f"CRITICAL: Anonymization pipeline failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Anonymization pipeline failed. Request aborted to prevent PII leakage. "
                f"Original error: {e}"
            )

    def get_pii_risk_level(self, result: AnonymizationResult) -> str:
        """
        Classifies the PII risk level of a message based on what was detected.
        This is a simple heuristic for research dashboard enrichment.

        RISK CLASSIFICATION LOGIC:
            - HIGH:   Identity documents or financial data detected (Aadhaar, PAN, etc.)
            - MEDIUM: Contact information detected (email, phone, location)
            - LOW:    Only soft PII detected (names, dates, ages)
            - NONE:   No PII detected

        Args:
            result (AnonymizationResult): The result from a prior anonymize() call.

        Returns:
            str: One of 'HIGH', 'MEDIUM', 'LOW', 'NONE'
        """
        if not result.pii_found:
            return "NONE"

        entities = set(result.detected_entities)

        # High-risk: Government IDs or financial identifiers
        high_risk_entities = {"IN_AADHAAR", "IN_PAN", "IN_VOTER", "IN_PASSPORT", "MEDICAL_LICENSE"}
        if entities & high_risk_entities:  # Set intersection — any overlap = HIGH risk
            return "HIGH"

        # Medium-risk: Direct contact information
        medium_risk_entities = {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "IP_ADDRESS"}
        if entities & medium_risk_entities:
            return "MEDIUM"

        # Low-risk: Soft identifiers (names, locations, dates)
        return "LOW"


# ==============================================================================
# SECTION 3: MODULE-LEVEL SINGLETON INSTANTIATION
# ==============================================================================
# ARCHITECTURE NOTE: We instantiate PrivacyEngine at MODULE IMPORT TIME.
# When main.py does `from privacy_engine import privacy_engine`, Python runs
# this line once and caches the module. Every subsequent import gets the same
# cached instance — effectively a Singleton.
#
# This means the expensive SpaCy model load happens ONCE at server startup,
# not on every API request. This is a critical performance optimization.
#
# The privacy_engine instance is what all other modules should import and use:
#   from backend.privacy_engine import privacy_engine
#   result = privacy_engine.anonymize("My name is Rahul")
# ==============================================================================
logger.info("Loading PrivacyEngine singleton at module import...")
privacy_engine = PrivacyEngine()


# ==============================================================================
# SECTION 4: STANDALONE TEST HARNESS
# ==============================================================================
# This block only runs when you execute this file directly:
#   python backend/privacy_engine.py
# It does NOT run when the file is imported by main.py.
# This is a Python convention (if __name__ == "__main__") that allows a module
# to serve dual purpose: importable library AND standalone test script.
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PrivacyEngine — Standalone Test Harness")
    print("="*60)

    # Test cases covering a range of PII types relevant to the Indian context.
    test_cases = [
        {
            "label": "No PII (should pass through unchanged)",
            "text":  "I have been feeling really anxious lately about my future."
        },
        {
            "label": "Person name + email + phone",
            "text":  "Hi, I'm Priya Sharma. You can reach me at priya.s@gmail.com or 9876543210."
        },
        {
            "label": "Location + date",
            "text":  "I live in Bangalore and was born on March 15, 1998."
        },
        {
            "label": "Indian Aadhaar number (HIGH risk)",
            "text":  "My Aadhaar number is 1234 5678 9012 and I feel very alone."
        },
        {
            "label": "Complex mixed PII",
            "text":  "My name is Arjun Mehta, I'm from Mumbai. Contact me at arjun@hotmail.com. "
                     "My PAN is ABCDE1234F. I've been struggling with depression since January 2023."
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {case['label']}")
        print(f"  INPUT:  {case['text']}")

        result = privacy_engine.anonymize(case["text"])
        risk   = privacy_engine.get_pii_risk_level(result)

        print(f"  OUTPUT: {result.anonymized_text}")
        print(f"  PII Found:  {result.pii_found}")
        print(f"  Entities:   {result.detected_entities}")
        print(f"  Count:      {result.entity_count}")
        print(f"  Risk Level: {risk}")
        print(f"  CSV:        {result.entities_csv}")

    print("\n" + "="*60)
    print("  All tests complete.")
    print("="*60 + "\n")