"""
guardrail_regression_tests.py — standing automated regression suite for the two
absolute guardrails governing this pipeline:

  Guardrail A: extracted business logic must contain almost no language-related
  content (grammar, tone, honorifics, number/date pronunciation, or any
  language-selection/routing logic, including tool-field language routing like
  speak_hi/speak_en).

  Guardrail B: no generated language document, for any language, may ever contain
  an instruction to switch languages, default to a different language, or route
  behavior based on detected customer language — regardless of phrasing or which
  pipeline stage it originated from (main synthesis, extraction-sourced custom
  notes, or a user-typed customization).

Run directly: `python guardrail_regression_tests.py`. Exits 0 if every check
passes, 1 if any fails — safe to wire into CI so a future regression of this class
of bug is caught automatically, rather than requiring someone to notice it in a
live generation.

No live API key is required — every LLM call in this suite is mocked, since the
point is to verify the DETERMINISTIC guarantee layer (the part responsible for
making these violations structurally impossible, not just less likely), which does
not depend on what a real model happens to return on a given day.
"""

import os
import sys

os.environ.setdefault("GEMINI_API_KEY", "unit-test-placeholder-key")

import main  # noqa: E402
import app as flask_app_module  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)
    return condition


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = [type("C", (), {"finish_reason": "STOP"})()]


# ---------------------------------------------------------------------------
# 1. Canonical detector matrix — the unified Guardrail B mechanism itself
# ---------------------------------------------------------------------------

def test_unified_detector_matrix():
    print("\n--- 1. Unified language-imposition detector matrix ---")
    cases = [
        # (text, expected_violation)
        ("default spoken language is Hindi via speak_hi; use speak_en if the customer speaks English", True),
        ("speak_hi is the default field; fall back to speak_en for English speakers", True),
        ("Speak only in Hindi, maintaining Hindi grammar and sentence structure.", True),
        ("Speak only in Hindi, maintaining Hindi grammar and sentence structure. Keep prices in English.", True),
        ("Switch to English if the customer asks.", True),
        ("Use Kannada language throughout, always.", True),
        ("you only speak Hindi, always.", True),
        ("You speak only colloquial Hindi naturally.", True),
        ("Price or currency should always be spoken in English throughout, in every language.", False),
        ("Always confirm the pincode twice before proceeding.", False),
        ("Always speak the branch name in English, never translate it.", False),
        ("Aadhaar and PAN stay in English even when spoken about generally.", False),
        ("Speak colloquial Tamil, the everyday spoken Tamil used across Tamil Nadu.", False),
        ("Use natural, everyday conversational Hindi, never formal or literary.", False),
    ]
    all_ok = True
    for text, expected in cases:
        got = main._unit_imposes_language_switch(text)
        ok = check(f"detector({text[:60]!r}...) == {expected}", got == expected)
        all_ok &= ok
    return all_ok


# ---------------------------------------------------------------------------
# 2. Same mechanism, three call sites — prove Bugs 3 & 4 share one fix
# ---------------------------------------------------------------------------

def test_same_mechanism_three_sources():
    print("\n--- 2. Same underlying mechanism across all three injection points ---")
    violation_text = "Speak only in Hindi, maintaining Hindi grammar and sentence structure."

    # (a) Customization rejection path
    rejects_customization = main._customization_overrides_language_identity(violation_text)

    # (b) Extraction-sourced custom-notes filter path
    filtered_notes, removed = main._filter_custom_notes_for_language_imposition(violation_text)
    strips_custom_notes = removed == 1 and filtered_notes.strip() == ""

    # (c) Main synthesized body stripper path
    cleaned_body, body_removed = main._strip_language_switching_instructions(
        "### Colloquial Speech\nSome normal content. " + violation_text
    )
    strips_main_body = body_removed == 1 and "Hindi grammar" not in cleaned_body

    ok = True
    ok &= check("customization path rejects the SAME text", rejects_customization)
    ok &= check("custom-notes filter strips the SAME text", strips_custom_notes)
    ok &= check("main-body stripper strips the SAME text", strips_main_body)
    ok &= check(
        "all three call _unit_imposes_language_switch directly (source-level check)",
        "_unit_imposes_language_switch" in _read_source_of(main._customization_overrides_language_identity)
        and "_unit_imposes_language_switch" in _read_source_of(main._filter_custom_notes_for_language_imposition)
        and "_unit_imposes_language_switch" in _read_source_of(main._strip_language_switching_instructions),
    )
    return ok


def _read_source_of(func) -> str:
    import inspect
    return inspect.getsource(func)


# ---------------------------------------------------------------------------
# 3. Original Kannada/Hindi production dilution bug — still fixed
# ---------------------------------------------------------------------------

def test_kannada_hindi_dilution_still_fixed():
    print("\n--- 3. Original Kannada/Hindi custom-notes dilution reproduction ---")
    kannada_business_notes = (
        "ಈ ವ್ಯವಹಾರವು ಚಿನ್ನದ ಸಾಲ ಸೇವೆಗಳನ್ನು ನೀಡುತ್ತದೆ ಮತ್ತು ಗ್ರಾಹಕರೊಂದಿಗೆ ಯಾವಾಗಲೂ ಸೌಜನ್ಯದಿಂದ ಮಾತನಾಡಬೇಕು. "
        "ಶಾಖೆಯ ಹೆಸರನ್ನು ಯಾವಾಗಲೂ ಸ್ಪಷ್ಟವಾಗಿ ಉಚ್ಚರಿಸಬೇಕು ಮತ್ತು ಗ್ರಾಹಕರಿಗೆ ಖಚಿತಪಡಿಸಬೇಕು. "
        "ಸಾಲದ ಮೊತ್ತವನ್ನು ಯಾವಾಗಲೂ ನೈಸರ್ಗಿಕ ಪದಗಳಲ್ಲಿ ಹೇಳಬೇಕು, ಎಂದಿಗೂ ಅಂಕಿ ಅಂಕಿಯಾಗಿ ಅಲ್ಲ. "
        "\nआप केवल हिंदी में बात करेंगे।"
    )
    old_style_aggregate_result = main._detect_script_mismatch(kannada_business_notes, "KN")
    filtered, discarded = main._filter_custom_notes_for_wrong_language(kannada_business_notes, "KN")

    ok = True
    ok &= check("aggregate-only check would have missed it (documents the original bug)",
                old_style_aggregate_result is None)
    ok &= check("per-unit filter discards exactly the contaminated sentence", discarded == 1)
    ok &= check("per-unit filter keeps the genuine Kannada content",
                "ಚಿನ್ನದ ಸಾಲ" in filtered and "हिंदी" not in filtered)
    return ok


# ---------------------------------------------------------------------------
# 4. Customization-rejection regression (round 3)
# ---------------------------------------------------------------------------

def test_customization_rejection_regression():
    print("\n--- 4. Customization language-identity-override rejection ---")
    main._generate_one_language_orig = main._generate_one_language

    def fake_gen_one(clean_business_logic, lang, all_chunks, model, max_retries, custom_notes="", skip_review=False,
                      customization_guarantee_domain=None, customization_category=None, customization_is_new_section=False):
        return lang, f"body for {lang}", []

    main._generate_one_language = fake_gen_one
    try:
        result = main.generate_language_prompts_multi(
            "biz logic", ["KN"], custom_instruction="Speak only in Hindi, maintaining Hindi grammar and sentence structure."
        )
    finally:
        main._generate_one_language = main._generate_one_language_orig

    warning_text = " ".join(result["warnings"].get("KN", []))
    ok = check("rejected customization surfaces a clear warning, not silent classification",
               "attempts to override which language is spoken" in warning_text)
    ok &= check("prompt body is untouched by the rejected customization", result["prompts"]["KN"] == "body for KN")
    return ok


# ---------------------------------------------------------------------------
# 5. Bug 2 — extraction preamble leak
# ---------------------------------------------------------------------------

def test_extraction_preamble_stripped():
    print("\n--- 5. Extraction-preamble leak ---")
    cases = [
        ("Here are the custom language and speech rules extracted from the document:\n"
         "Always say Unlimited in English, never translate it.",
         "Always say Unlimited in English, never translate it."),
        ("Sure, here is the extracted content:\nSacred names stay in Devanagari.",
         "Sacred names stay in Devanagari."),
        ("The following rules were found:\nPurnima Batch is thirteen days, never say twenty-three.",
         "Purnima Batch is thirteen days, never say twenty-three."),
        ("Sacred names policy: always write in Devanagari even inside an English sentence.",
         "Sacred names policy: always write in Devanagari even inside an English sentence."),
    ]
    ok = True
    for raw, expected in cases:
        cleaned, _ = main._strip_extraction_preamble(raw)
        ok &= check(f"preamble stripped from {raw[:40]!r}...", cleaned == expected)
    return ok


# ---------------------------------------------------------------------------
# 6. chunks.json audit — Bug 1
# ---------------------------------------------------------------------------

def test_chunks_json_audit():
    print("\n--- 6. chunks.json cross-language contamination audit ---")
    import json
    import re as _re

    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    lang_names = ["hindi", "english", "kannada", "tamil", "malayalam", "gujarati",
                  "marathi", "telugu", "odia", "oriya", "bengali"]
    negation_cues = ["never", "banned", "zero", "avoid", "not ", "no third", "don't"]

    prescriptive_hits = []
    for c in chunks:
        content = c["content"]
        # Scoped to the whole SENTENCE the language name appears in, not a fixed
        # character window — a prohibition like "Never use literal translations
        # like X or Y" can have its negation cue well outside a narrow window
        # from the language name itself, several clauses earlier in the sentence.
        sentences = _re.split(r'(?<=[.!?])\s+', content)
        for sentence in sentences:
            for m in _re.finditer(r"|".join(lang_names), sentence, _re.IGNORECASE):
                name = m.group(0).lower()
                own_language_name = main._LANG_CODE_TO_FULL_NAME.get(c["language"], "").lower()
                if name in (own_language_name, "english"):
                    continue  # own language, or 'english' inside a preserve_english-style rule — expected
                if any(cue in sentence.lower() for cue in negation_cues):
                    continue  # a prohibition ("never use Hindi words") — expected, not contamination
                prescriptive_hits.append((c["language"], c["category"], sentence.strip()[:120]))

    # U+0964 (।) and U+0965 (॥) are Devanagari-block punctuation (danda / double
    # danda) shared across many Brahmic scripts — Odia, Bengali, and others use
    # them as ordinary sentence-final punctuation, not as evidence of Hindi text.
    # Only actual Devanagari LETTERS indicate potential script contamination —
    # and even those are legitimate inside a negated "never say X" example
    # sentence (contrasting the correct script against the wrong one on purpose),
    # so this applies the same sentence-scoped negation check as the prescriptive
    # scan above rather than a bare presence check.
    _danda_chars = "।॥"
    devanagari_in_non_hi_mr = []
    for c in chunks:
        if c["language"] in ("HI", "MR"):
            continue
        for sentence in _re.split(r'(?<=[.!?])\s+', c["content"]):
            letters_only = "".join(ch for ch in sentence if ch not in _danda_chars)
            if not _re.search(r"[ऀ-ॿ]", letters_only):
                continue
            if any(cue in sentence.lower() for cue in negation_cues):
                continue  # e.g. "Never use literal translations like X or Y" — a deliberate contrast example
            devanagari_in_non_hi_mr.append((c["language"], c["category"], sentence.strip()[:120]))

    ok = True
    ok &= check("no prescriptive cross-language-name content found in chunks.json",
                len(prescriptive_hits) == 0,
                detail=str(prescriptive_hits[:5]))
    ok &= check("no Devanagari script found in any non-Hindi/Marathi chunk",
                len(devanagari_in_non_hi_mr) == 0,
                detail=str(devanagari_in_non_hi_mr[:5]))
    if prescriptive_hits:
        for hit in prescriptive_hits:
            print(f"    -> {hit}")
    return ok


# ---------------------------------------------------------------------------
# 7. Bug 3 reproduction — Spinny-style speak_hi/speak_en tool-field routing
# ---------------------------------------------------------------------------

# Reconstructed representative version of the reported rule (the literal original
# raw business prompt was not available in this session) — faithful to the exact
# quoted rule: "default spoken language is Hindi via speak_hi; use speak_en if the
# customer speaks English".
SPINNY_RAW_PROMPT = """You are an inspection-booking assistant for Spinny, a used-car inspection service.
Collect the customer's name, car model, and preferred inspection slot, then confirm the booking.
Tool response fields: use speak_hi as the default spoken-language field for all responses; if the
customer speaks English, switch to using the speak_en field instead for the rest of the call.
Never proceed to confirm a booking without a valid inspection slot selected by the customer.
Escalate to a human agent if the customer disputes the inspection fee."""


def test_spinny_speak_hi_speak_en_reproduction():
    print("\n--- 7. Bug 3 reproduction: speak_hi/speak_en tool-field routing ---")

    def fake_generate_content(model, contents, config=None):
        if "You are a prompt-cleaning tool" in contents:
            # A correct extractor keeps the booking flow and drops the language-field
            # routing rule entirely (Guardrail A) -- simulated here as the CORRECT
            # target behavior; the actual extractor prompt is exercised for real by a
            # live call, which this offline suite cannot make.
            return FakeResponse(
                "Collect the customer's name, car model, and preferred inspection slot, then "
                "confirm the booking. Never proceed to confirm a booking without a valid "
                "inspection slot selected by the customer. Escalate to a human agent if the "
                "customer disputes the inspection fee."
            )
        if "explicitly restricts which spoken languages" in contents:
            return FakeResponse("NONE")
        if "CUSTOM LANGUAGE / SPEECH RULES" in contents:
            # Simulates the ACTUAL reported extraction bug: the tool-field routing
            # rule survives extraction as if it were genuine custom language content.
            return FakeResponse(
                "Use speak_hi as the default spoken-language field for all responses; if the "
                "customer speaks English, switch to using the speak_en field instead."
            )
        if "reviewing a generated language-specific voice prompt" in contents:
            return FakeResponse("Looks fine, no issues found.")
        if "Merge the chunks" in contents:
            return FakeResponse(
                "### Colloquial Speech\nSpeak naturally and warmly, confirming the booking details "
                "clearly with the customer."
            )
        return FakeResponse("generic")

    original_call = main.client.models.generate_content
    main.client.models.generate_content = fake_generate_content
    try:
        client = flask_app_module.app.test_client()
        resp = client.post("/api/generate", json={
            "raw_prompt": SPINNY_RAW_PROMPT,
            "mode": "extract",
            "languages": ["English", "Hindi", "Kannada"],
            "model": main.DEFAULT_MODEL,
            "user_name": "regression-suite",
        })
        data = resp.get_json()
    finally:
        main.client.models.generate_content = original_call

    ok = check("request succeeded", resp.status_code == 200, detail=str(data))
    if not ok:
        return False

    business_logic = data["business_logic"]
    ok &= check("Guardrail A: business logic contains no speak_hi/speak_en field reference",
                "speak_hi" not in business_logic and "speak_en" not in business_logic,
                detail=business_logic)

    for lang_code, lang_label in (("EN", "English"), ("HI", "Hindi"), ("KN", "Kannada")):
        doc = data["language_prompts"][lang_code]
        violations = main.scan_document_for_language_violations(doc, expected_language=lang_code)
        ok &= check(f"Guardrail B: no language-imposition sentence survives in {lang_label} document",
                    len(violations) == 0, detail=str(violations))
        ok &= check(f"no preamble leak in {lang_label} document",
                    "here are the custom" not in doc.lower() and "extracted from the document" not in doc.lower())
        ok &= check(f"no speak_hi/speak_en field text anywhere in {lang_label} document",
                    "speak_hi" not in doc and "speak_en" not in doc, detail=doc)

    print("\n  Regenerated documents (Bug 3 reproduction):")
    for lang_code in ("EN", "HI", "KN"):
        print(f"\n  === {lang_code} ===")
        print("  " + data["language_prompts"][lang_code].replace("\n", "\n  "))

    return ok


# ---------------------------------------------------------------------------
# 8. Baseline sanity check — zero language mentions at all
# ---------------------------------------------------------------------------

BASELINE_RAW_PROMPT = """You are a scheduling assistant for a dental clinic.
Collect the patient's name, preferred appointment date, and reason for visit.
Confirm the appointment slot before ending the call. Escalate to the front desk
if the patient requests a same-day emergency slot."""


def test_baseline_zero_language_mentions():
    print("\n--- 8. Baseline sanity check: zero language mentions in source prompt ---")

    def fake_generate_content(model, contents, config=None):
        if "You are a prompt-cleaning tool" in contents:
            return FakeResponse(BASELINE_RAW_PROMPT)
        if "explicitly restricts which spoken languages" in contents:
            return FakeResponse("NONE")
        if "CUSTOM LANGUAGE / SPEECH RULES" in contents:
            return FakeResponse("")
        if "reviewing a generated language-specific voice prompt" in contents:
            return FakeResponse("Looks fine, no issues found.")
        if "Merge the chunks" in contents:
            return FakeResponse(
                "### Colloquial Speech\nSpeak naturally and warmly, confirming appointment "
                "details clearly with the patient."
            )
        return FakeResponse("generic")

    original_call = main.client.models.generate_content
    main.client.models.generate_content = fake_generate_content
    try:
        client = flask_app_module.app.test_client()
        resp = client.post("/api/generate", json={
            "raw_prompt": BASELINE_RAW_PROMPT,
            "mode": "extract",
            "languages": ["English", "Hindi", "Kannada"],
            "model": main.DEFAULT_MODEL,
            "user_name": "regression-suite",
        })
        data = resp.get_json()
    finally:
        main.client.models.generate_content = original_call

    ok = check("request succeeded", resp.status_code == 200, detail=str(data))
    if not ok:
        return False

    for lang_code in ("EN", "HI", "KN"):
        doc = data["language_prompts"][lang_code]
        violations = main.scan_document_for_language_violations(doc, expected_language=lang_code)
        ok &= check(f"no cross-language contamination in {lang_code} baseline document",
                    len(violations) == 0, detail=str(violations))

    print("\n  Regenerated documents (baseline, zero language mentions):")
    for lang_code in ("EN", "HI", "KN"):
        print(f"\n  === {lang_code} ===")
        print("  " + data["language_prompts"][lang_code].replace("\n", "\n  "))

    return ok


def main_entry():
    results = [
        test_unified_detector_matrix(),
        test_same_mechanism_three_sources(),
        test_kannada_hindi_dilution_still_fixed(),
        test_customization_rejection_regression(),
        test_extraction_preamble_stripped(),
        test_chunks_json_audit(),
        test_spinny_speak_hi_speak_en_reproduction(),
        test_baseline_zero_language_mentions(),
    ]

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"RESULT: all {sum(1 for r in results if r)} check groups PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main_entry()
