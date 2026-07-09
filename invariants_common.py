"""
invariants_common.py

Single source of truth for the protected-invariant registry and the pattern-check
logic, shared by:
  - chunk_lint.py       (checks chunks.json at authoring time)
  - main.py             (checks the FINAL generated output at generation time)

Two checkpoints matter because they catch different failure modes:
  - chunks.json can be clean, but the synthesizer LLM can still paraphrase a
    rule out of existence while merging chunks — that's only visible in the
    output, not the source.
  - Conversely, checking only the output can't tell you WHERE a bad rule
    originated if two chunks contradict each other before synthesis even runs.
"""

import os
import json
import re

# ---- Hardcoded starter list — used only until you build a full invariants.json
# from your own chunks.json via invariant_extractor.py. ----

PROTECTED_INVARIANTS = [
    {
        "name": "pincode_digit_by_digit",
        "applies_to_tags": {"pincode"},
        "forbidden_patterns": [
            r"read.{0,20}pin.{0,20}(as a|as one|whole)\s*number",
            r"pin\s*code.{0,30}(natural|spoken)\s*number\s*word",
        ],
        "must_contain_any": [r"digit[\s-]?by[\s-]?digit"],
        "message": "PIN codes must always be confirmed digit-by-digit — never as a natural spoken number or whole value.",
    },
    {
        "name": "phone_number_digit_by_digit",
        "applies_to_tags": {"phone_number"},
        "forbidden_patterns": [r"phone.{0,30}(natural|spoken)\s*number\s*word"],
        "must_contain_any": [r"digit[\s-]?by[\s-]?digit"],
        "message": "Phone/mobile numbers must always be read digit-by-digit — never as a natural spoken number.",
    },
    {
        # PROXIMITY-SCOPED: only flags digit-by-digit if it appears near a
        # currency/amount keyword. Previously this matched "digit-by-digit"
        # ANYWHERE in the whole output (e.g. in an unrelated PIN section),
        # producing false positives even when the currency section itself
        # was correctly written.
        "name": "currency_natural_words_not_digits",
        "applies_to_tags": {"currency"},
        "forbidden_patterns": [
            r"(rupee|currency|amount|price|fee|cost)\w*.{0,60}digit[\s-]?by[\s-]?digit",
            r"digit[\s-]?by[\s-]?digit.{0,60}(rupee|currency|amount|price|fee|cost)",
        ],
        "must_contain_any": [r"natural|spoken|word"],
        "message": "Currency/loan amounts must be read as natural spoken words, never digit-by-digit.",
    },
    {
        # PROXIMITY-SCOPED for the same reason as currency above.
        "name": "gold_weight_natural_words",
        "applies_to_tags": {"gold_weight"},
        "forbidden_patterns": [
            r"(gold|weight|gram)\w*.{0,60}digit[\s-]?by[\s-]?digit",
            r"digit[\s-]?by[\s-]?digit.{0,60}(gold|weight|gram)",
        ],
        "must_contain_any": [r"natural|spoken|word"],
        "message": "Gold weight must be read as natural spoken words, never digit-by-digit.",
    },
    {
        # Disabled by default (must_contain_any left empty) — this rule assumed
        # every business needs weekday-paired dates, which isn't true for all
        # use cases (e.g. Muthoot's date chunks never mention weekday pairing
        # and don't need to). Fill must_contain_any back in only if your
        # business logic actually requires weekday pairing for dates.
        "name": "dates_paired_with_weekday",
        "applies_to_tags": {"dates"},
        "forbidden_patterns": [],
        "must_contain_any": [],
        "message": "Any rule governing how dates are spoken should require pairing the date with its weekday.",
    },
]


def load_invariants(invariants_file: str = "invariants.json") -> list:
    """Loads the reviewed registry if it exists; falls back to the hardcoded
    starter list otherwise. Normalizes applies_to_tags to a set either way."""
    if os.path.exists(invariants_file):
        with open(invariants_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        normalized = []
        for rule in loaded:
            if rule.get("reviewed") is False:
                continue
            normalized.append({
                "name": rule.get("name", "unnamed"),
                "applies_to_tags": set(rule.get("applies_to_tags", [])),
                "forbidden_patterns": rule.get("forbidden_patterns", []),
                "must_contain_any": rule.get("must_contain_any", []),
                "message": rule.get("description", ""),
            })
        return normalized

    return [{**rule, "applies_to_tags": set(rule["applies_to_tags"])} for rule in PROTECTED_INVARIANTS]


def check_text_against_invariants(text: str, tags: set, invariants: list) -> list:
    """Runs forbidden/must-contain checks against a block of TEXT (either a chunk's
    content, or a fully synthesized output) for whichever invariants apply to `tags`.
    Returns a list of human-readable violation messages."""
    text_lower = text.lower()
    violations = []

    for rule in invariants:
        if not (tags & rule["applies_to_tags"]):
            continue

        for pattern in rule.get("forbidden_patterns", []):
            if re.search(pattern, text_lower, flags=re.DOTALL):
                violations.append(f"violates '{rule['name']}': matched forbidden pattern /{pattern}/ — {rule['message']}")

        must_contain_any = rule.get("must_contain_any", [])
        if must_contain_any and len(text) > 40 and not any(re.search(p, text_lower) for p in must_contain_any):
            violations.append(f"possibly missing '{rule['name']}': expected one of {must_contain_any} — {rule['message']}")

    return violations