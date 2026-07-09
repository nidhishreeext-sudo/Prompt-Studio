import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

from invariants_common import load_invariants, check_text_against_invariants

load_dotenv()

_INVARIANTS = load_invariants()  # loaded once; reflects invariants.json if you've built one, else the starter list
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- MODEL CONFIG ----------

SUPPORTED_MODELS = {
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "supports_thinking_level": False,
    },
    "gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash",
        "supports_thinking_level": True,
    },
}

DEFAULT_MODEL = "gemini-2.5-flash"


def _resolve_model(model: str | None) -> str:
    if model in SUPPORTED_MODELS:
        return model
    return DEFAULT_MODEL


def _build_config(model: str, max_output_tokens: int | None = None, thinking_level: str = "low") -> dict:
    """Build a GenerateContentConfig dict, only attaching thinking_config for models that support it."""
    config = {}
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens

    if SUPPORTED_MODELS[model]["supports_thinking_level"]:
        config["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    return config


# ---------- STAGE 1: Business Logic Extractor ----------

EXTRACTOR_SYSTEM_PROMPT = """You are a prompt-cleaning tool. You will be given a raw AI agent system prompt that mixes business logic (conversation flow, node structure, tool calls, guardrails, knowledge base) with language-specific rules (grammar, gender forms, colloquial speech, honorifics, number/currency/date formatting, backchannels, fillers, pronunciation rules, script rules).

Your job: output ONLY the business logic. Remove every language-specific rule, grammar rule, gender-form rule, colloquial-speech instruction, honorific rule, pronunciation rule, and formatting-of-numbers/dates/currency-in-speech rule.

Keep: conversation flow/nodes, tool invocation logic, guardrails unrelated to language, knowledge base facts, eligibility rules, appointment validation logic, objection handling logic (the WHAT to say, not the HOW in a specific language).

Do NOT change, reorder, or summarize the business logic itself — only remove language-specific content. Output the cleaned prompt only, no commentary."""


def extract_business_logic(raw_prompt: str, model: str = DEFAULT_MODEL) -> str:
    model = _resolve_model(model)
    full_prompt = f"{EXTRACTOR_SYSTEM_PROMPT}\n\n---RAW PROMPT---\n{raw_prompt}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model)
    )
    return response.text


# ---------- STAGE 3: Relevance Matcher ----------
TAG_TRIGGERS = {
    "pincode": ["pin code", "pincode", "postal code"],
    "phone_number": ["phone number", "mobile number", "contact number"],
    "currency": ["rupee", "loan amount", "₹", "rate", "interest", "price", "fee", "cost", "amount"],
    "gold_weight": ["gold weight", "grams", "gold quantity", "tola", "kg", "gold"],
    "dates": ["appointment", "visit day", "callback", "schedule", "date"],
    "branch_names": ["branch", "location", "nearest"],
}

ALWAYS_ON_TAGS = ["colloquial", "honorifics", "agent_gender", "call_opening", "call_closing",
                   "backchannels", "fillers", "numbers_general", "escalation", "sensitive_situation"]
def match_relevant_chunks(business_logic: str, language: str, all_chunks: list) -> list:
    business_logic_lower = business_logic.lower()
    triggered_tags = set()

    for tag, keywords in TAG_TRIGGERS.items():
        if any(kw in business_logic_lower for kw in keywords):
            triggered_tags.add(tag)

    triggered_tags.update(ALWAYS_ON_TAGS)

    matched = [
        chunk for chunk in all_chunks
        if chunk["language"] == language
        and any(tag in triggered_tags for tag in chunk["tags"])
    ]
    return matched


# ---------- STAGE 4: Synthesizer ----------

SYNTHESIZER_SYSTEM_PROMPT = """You are given a business-logic system prompt for a voice AI agent, and a set of relevant detailed language-specific rule chunks for a target language.

Merge the chunks into ONE comprehensive, well-organized language prompt. Be thorough, not brief — this is a production specification, not a summary.

CRITICAL REQUIREMENTS:
- Preserve every concrete example (✓/✗ pairs, sample sentences, tables) found in the chunks EXACTLY as given. Do not drop examples to save space.
- Preserve every specific rule, exception, and edge case mentioned in the chunks — do not compress multiple distinct rules into one vague sentence.
- If a chunk contains grammar rules (suffixes, verb forms, postpositions), reproduce the full rule set with its examples, not just a one-line summary.
- Organize by clear category headers (Colloquial Speech, Grammar, Honorifics, Numbers, Currency, Backchannels, Fillers, Call Flow, etc.)
- Remove only exact duplicate content across chunks. Do not remove detail, examples, or nuance in the name of conciseness.
- Do not include any business logic — only language/speech rules.
- Length should reflect the depth of the source chunks. A rich set of chunks should produce a long, detailed prompt — do not artificially shorten it.

STRICT NO-INVENTION RULE (applies to every category, especially Fillers and Backchannels):
- Use ONLY the words, phrases, and examples that literally appear in the provided chunks below. Never invent, guess, or supplement with additional filler words, backchannel phrases, honorific forms, or example sentences that are not present in the source chunks — even if you believe them to be correct or natural for this language.
- If a chunk's coverage for a category is thin (e.g. only two filler words given), output only those — do not pad the list with more of your own to seem more "thorough." Thoroughness means fully using what's given, not extending it.
- If a category (e.g. Backchannels) has no corresponding chunk at all in the input below, omit that section entirely rather than generating one from general knowledge of the language.

Output the final language prompt only, no commentary."""


def synthesize_language_prompt(business_logic: str, relevant_chunks: list, language: str, model: str = DEFAULT_MODEL) -> str:
    model = _resolve_model(model)
    chunks_text = "\n\n".join([f"[{c['category']}]\n{c['content']}" for c in relevant_chunks])

    full_prompt = f"""{SYNTHESIZER_SYSTEM_PROMPT}

BUSINESS LOGIC CONTEXT (for relevance only, do not include in output):
{business_logic[:2000]}

RELEVANT LANGUAGE CHUNKS FOR {language}:
{chunks_text}"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=4000)
    )
    return response.text


# ---------- FULL PIPELINE ----------

def generate_language_prompt(raw_prompt: str, language: str, chunks_file="chunks.json", model: str = DEFAULT_MODEL) -> dict:
    biz_logic = extract_business_logic(raw_prompt, model=model)

    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    relevant_chunks = match_relevant_chunks(biz_logic, language, all_chunks)
    final_language_prompt = synthesize_language_prompt(biz_logic, relevant_chunks, language, model=model)

    return {
        "business_logic": biz_logic,
        "language_prompt": final_language_prompt
    }


def generate_language_prompts_multi(clean_business_logic: str, languages: list, chunks_file="chunks.json", model: str = DEFAULT_MODEL) -> dict:
    """Mode 2: business logic is already clean — skip extraction, generate scoped language prompts for multiple languages.

    Returns {"prompts": {lang: text}, "warnings": {lang: [violation strings]]}}.
    Warnings come from checking the FINAL synthesized text against invariants.json —
    this is the checkpoint that catches the synthesizer paraphrasing away a solved
    rule (e.g. pincode digit-by-digit) even when chunks.json itself was clean and
    conflict-free. chunk_lint.py alone can't see this; it only ever reads chunks.json,
    never the model's actual merged output.
    """
    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    prompts = {}
    warnings = {}

    for lang in languages:
        relevant_chunks = match_relevant_chunks(clean_business_logic, lang, all_chunks)
        output_text = synthesize_language_prompt(clean_business_logic, relevant_chunks, lang, model=model)
        prompts[lang] = output_text

        # Union of tags actually pulled in for this language — only check invariants
        # relevant to what was actually supposed to be covered.
        triggered_tags = set()
        for chunk in relevant_chunks:
            triggered_tags.update(chunk.get("tags", []))

        lang_warnings = check_text_against_invariants(output_text, triggered_tags, _INVARIANTS)
        if lang_warnings:
            warnings[lang] = lang_warnings

    return {"prompts": prompts, "warnings": warnings}


LANGUAGE_NAME_MAP = {
    "hindi": "HI", "hi": "HI",
    "english": "EN", "en": "EN",
    "kannada": "KN", "kannad": "KN", "kn": "KN",
    "tamil": "TA", "ta": "TA",
    "malayalam": "ML", "ml": "ML",
    "gujarati": "GU", "gujrati": "GU", "gu": "GU",
    "marathi": "MR", "mr": "MR",
    "telugu": "TE", "te": "TE",
}


def resolve_languages(user_input: list) -> list:
    """Convert user-typed language names/codes into internal codes, case-insensitive, dedup, preserve order."""
    resolved = []
    for name in user_input:
        code = LANGUAGE_NAME_MAP.get(name.strip().lower())
        if code and code not in resolved:
            resolved.append(code)
        elif not code:
            print(f"⚠️ Unrecognized language: '{name}' — skipped")
    return resolved


if __name__ == "__main__":
    clean_business_logic = """# System Mandates and Internal Context
(... same as before, truncated in this refactor for brevity — CLI test block unchanged ...)"""

    user_requested = ["Gujarati", "Kannada", "Tamil"]
    languages = resolve_languages(user_requested)

    result = generate_language_prompts_multi(clean_business_logic, languages)

    os.makedirs("output", exist_ok=True)
    with open("output/business_logic.txt", "w", encoding="utf-8") as f:
        f.write(clean_business_logic)

    for lang, prompt in result["prompts"].items():
        with open(f"output/language_prompt_{lang}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        warn_count = len(result["warnings"].get(lang, []))
        flag = f" — ⚠ {warn_count} invariant warning(s)" if warn_count else ""
        print(f"✓ {lang} done — {len(prompt)} chars{flag}")

    if result["warnings"]:
        print("\nInvariant warnings:")
        for lang, warns in result["warnings"].items():
            for w in warns:
                print(f"  [{lang}] {w}")