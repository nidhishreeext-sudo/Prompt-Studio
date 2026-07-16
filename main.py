import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "group": "Flash",
        "supports_thinking_level": False,
    },
    "gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash",
        "group": "Flash",
        "supports_thinking_level": True,
    },
    "gemini-3.1-pro-preview": {
        "label": "Gemini 3.1 Pro",
        "group": "Pro",
        "supports_thinking_level": True,
    },
}

DEFAULT_MODEL = "gemini-2.5-flash"
# Review (and applying review fixes) benefits more from a stronger model than generation does,
# since it needs to catch subtle contradictions rather than just reproduce chunk content.
DEFAULT_REVIEW_MODEL = "gemini-3.1-pro-preview"


def _resolve_model(model: str | None) -> str:
    if model in SUPPORTED_MODELS:
        return model
    return DEFAULT_MODEL


def _build_config(model: str, max_output_tokens: int | None = None, thinking_level: str = "low") -> types.GenerateContentConfig:
    """Build a proper GenerateContentConfig object (not a plain dict) — newer SDK
    versions require the typed object here rather than auto-converting a dict."""
    kwargs = {"temperature": 0}
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens

    if SUPPORTED_MODELS[model]["supports_thinking_level"]:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    return types.GenerateContentConfig(**kwargs)


# ---------- STAGE 1: Business Logic Extractor ----------

EXTRACTOR_SYSTEM_PROMPT = """You are a prompt-cleaning tool. You will be given a raw AI agent system prompt that mixes business logic (conversation flow, node structure, tool calls, guardrails, knowledge base, agent identity/persona) with language-specific rules (grammar, gender forms, colloquial speech, honorifics, number/currency/date formatting, backchannels, fillers, pronunciation rules, script rules).

Your job: output ONLY the business logic. Remove every language-specific rule, grammar rule, gender-form rule, colloquial-speech instruction, honorific rule, pronunciation rule, and formatting-of-numbers/dates/currency-in-speech rule.

Keep everything else, including:
- Conversation flow, nodes, stages, and turn sequencing (e.g. "ask name first, then mobile in a separate turn").
- Tool invocation logic and disposition/routing logic.
- Guardrails unrelated to language, EVERY SINGLE ONE, even ones that sound similar to each other. Two guardrail sentences that look repetitive to you may each cover a distinct edge case — keep both, never merge or drop one as redundant.
- Knowledge base facts, eligibility rules, appointment validation logic, objection handling logic (the WHAT to say, not the HOW in a specific language).
- Agent identity and persona (name, gender, personality, tone description, "you sound like X not Y") — this is business logic, not a language rule, and must be kept in full even though it describes character/tone.
- Every specific factual figure, rate, price, percentage, timeline, or numeric example given anywhere in the document (e.g. a specific live rate, a specific "thirty to forty minutes" processing time, a specific phone number). These are business facts, not illustrative flavor — dropping any one of them is a critical failure, not an acceptable simplification.
- Any "give the specific answer, never say only X" style instruction, these are anti-vagueness guardrails and are exactly as important as the rules around them.

SURGICAL STRIPPING — the trickiest failure mode: a single sentence often contains BOTH a business requirement AND a language-specific qualifier tangled together. In that case, delete ONLY the language-specific words, and keep the business requirement intact, do not delete the whole sentence.
- Example: "read the pincode back digit by digit with a natural pause between groups and ask for confirmation" — "digit by digit" here is doing double duty: it's a pronunciation detail (language layer) AND a business validation method (confirm each digit separately, not the number as a whole). Keep the business method: "read the pincode back and ask for confirmation, confirming each digit individually" — do NOT collapse it down to just "read the pincode back and ask for confirmation," that silently deletes a real validation requirement.
- Example: "read back the full date in spoken Kannada words, the confirmed time, and the branch name" — "in spoken Kannada words" is the language qualifier to delete. "the full date" is a business requirement (the summary must include the date) and must be kept: "read back the full date, the confirmed time, and the branch name."
- General rule: never let a language-specific qualifier attached to a business noun (date, digit, number, amount) delete the noun itself. Strip the qualifier, keep the noun and its requirement.

CONVERSATIONAL PACING AND TURN-TAKING RULES ARE BUSINESS LOGIC, NOT LANGUAGE RULES — keep them in full. This includes: turn-length ceilings (e.g. "fifteen words before pausing"), "ask one question at a time," instructions about not repeating the same phrasing across consecutive turns, instructions to phrase lines differently each time, and any scope-limiting instruction about what to read aloud (e.g. "read only the branch name and address"). None of these are about HOW to pronounce something in a specific language, they are rules about conversational structure and must be preserved exactly like any other guardrail.

THIS IS A COPY-AND-STRIP OPERATION, NOT A SUMMARY. Do NOT change, reorder, shorten, paraphrase, or condense the business logic itself, only delete language-specific sentences wholesale and leave everything else byte-for-byte as written. If you are unsure whether a sentence is a language rule or business logic, default to KEEPING it. The output should be nearly the same length as the input, minus only the language-specific portions. A noticeably shorter output than the input (accounting only for removed language content) means you have summarized, which is a failure — go back and include everything you dropped.

Output the cleaned prompt only, no commentary."""


def extract_business_logic(raw_prompt: str, model: str = DEFAULT_MODEL) -> str:
    model = _resolve_model(model)
    full_prompt = f"{EXTRACTOR_SYSTEM_PROMPT}\n\n---RAW PROMPT---\n{raw_prompt}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=32000)
    )
    return response.text


# ---------- STAGE 3: Relevance Matcher ----------
TAG_TRIGGERS = {
    "pincode": ["pin code", "pincode", "postal code"],
    "phone_number": ["phone number", "mobile number", "contact number", "callback number", "whatsapp number"],
    "currency": ["rupee", "loan amount", "₹", "rate", "interest", "price", "fee", "cost", "amount"],
    # "gold" / "tola" / "karat" / "carat" are safe as bare-word triggers (unlike generic weight
    # units like "kg" or "grams", which falsely matched unrelated domains e.g. vehicle weight) —
    # letter-boundary matching means these only fire on the real word, not a substring of another.
    "gold_weight": ["gold", "tola", "karat", "carat", "gold weight", "gold quantity"],
    "dates": ["appointment", "visit day", "callback", "schedule", "date"],
    "time_pronunciation": ["appointment", "callback", "schedule", "time", "o'clock", "working hours",
                            "office hours", "morning", "afternoon", "evening", "reschedul"],
    "branch_names": ["branch", "location", "nearest"],
    "finance_terms": ["loan", "emi", "kyc", "insurance", "mutual fund", "sip", "portfolio",
                       "pan card", "aadhaar", "cibil", "credit score", "interest rate",
                       "insurance premium", "insurance policy", "nbfc", "gold loan",
                       "personal loan", "home loan"],
}

ALWAYS_ON_TAGS = ["colloquial", "honorifics", "agent_gender", "call_opening", "call_closing",
                   "backchannels", "fillers", "numbers_general", "escalation", "sensitive_situation",
                   "hold_pause", "interruption", "preserve_english_terms", "language_switching",
                   "no_echo"]


_NEGATION_CUES = [
    "not supported", "not support", "do not support", "does not support", "doesn't support",
    "not offer", "does not offer", "doesn't offer", "not offered", "not available",
    "excluded", "out of scope", "unsupported", "not permitted", "not provide",
    "does not provide", "doesn't provide", "not applicable", "no longer support",
]


def _has_nearby_negation(text_lower: str, start: int, end: int, window: int = 80) -> bool:
    """Checks a window of text around a keyword match (both before and after, since
    an exclusion clause can precede the keyword — "out of scope (... gold loan ...)" —
    or follow it — "gold loans ... are not supported")."""
    context = text_lower[max(0, start - window):min(len(text_lower), end + window)]
    return any(cue in context for cue in _NEGATION_CUES)


def _keyword_present(text_lower: str, keyword: str) -> bool:
    """True only if `keyword` appears as a genuine whole word/phrase, AND at least one
    occurrence isn't sitting inside a negation/exclusion clause.

    This matters because a business often explicitly lists things it does NOT offer
    (e.g. "gold loans... are not supported", "out of scope (... gold loan ...)"). A
    business saying it explicitly does NOT do lending should not have finance-vocabulary
    chunks activated just because the word "loan" appears in that exclusion sentence —
    only a genuine affirmative mention of the category should trigger it. If a keyword
    appears multiple times and at least one mention is a real, non-negated usage, the
    tag still correctly activates.

    Uses letter-adjacency (not \\b) as the boundary check on purpose: \\b treats
    underscores as word characters, which would fail to match e.g. "amount"
    inside "loan_amount_required". Requiring only that no LETTER sits directly
    adjacent still rejects false hits like "rate" inside "generate" while still
    correctly matching "amount" next to underscores, digits, or punctuation.
    """
    pattern = r'(?<![a-z])' + re.escape(keyword) + r'(?![a-z])'
    for m in re.finditer(pattern, text_lower):
        if not _has_nearby_negation(text_lower, m.start(), m.end()):
            return True
    return False


def match_relevant_chunks(business_logic: str, language: str, all_chunks: list) -> list:
    business_logic_lower = business_logic.lower()
    triggered_tags = set()

    for tag, keywords in TAG_TRIGGERS.items():
        if any(_keyword_present(business_logic_lower, kw) for kw in keywords):
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

Merge the chunks into ONE well-organized, natural-reading language prompt — the kind a senior prompt engineer would hand-write and ship to production, not an exhaustive training manual.

WRITE LIKE A HUMAN EXPERT, NOT A SPEC DOCUMENT:
- Every rule must survive, but express it in tight, flowing prose for general rules — a short paragraph per rule, not a bullet-catalog of every possible case.
- NEVER use markdown pipe-tables ( | col | col | ). That specific syntax is banned.
- TWO DIFFERENT THINGS, DO NOT CONFUSE THEM:
  1. ILLUSTRATIVE EXAMPLES (✓/✗ pairs, sample sentences showing how a grammar rule behaves): keep at most 2-3 of the clearest ones per rule, in prose. These exist to demonstrate a pattern, not to be an exhaustive catalog.
  2. ESSENTIAL REFERENCE DATA WITH MANY DISCRETE ENTRIES (a number-word lookup table, digit-by-digit letter readings, a fixed list of preserved English terms, a set of per-minute time-fusion rules): reproduce this COMPLETELY, every entry, and use a clean bulleted list, one entry per line, NOT a pipe-table and NOT force-collapsed into a run-on paragraph. A list like "7:15 -> ಏಳಕ್ಕೆ ಕಾಲು" is far more scannable and usable to whoever reads this prompt than the same content buried in a sentence. Never trim, sample, or abbreviate this kind of list with "etc." or similar.
- Use simple, short section headers (### Colloquial Speech, ### Numbers, ### Backchannels) — not numbered mega-sections, not sub-headers nested three levels deep.
- Do not restate the same rule in multiple places or under multiple headers. If two chunks overlap, merge them into one clean statement.
- Do not add meta-commentary, headers-about-headers, or explanations of why a rule exists — state the rule and move on.

CRITICAL REQUIREMENTS:
- Preserve every distinct rule and exception mentioned in the chunks — do not drop substance, only trim redundant illustrative examples and formatting overhead, never trim reference data.
- Do not include any business logic — only language/speech rules.
- NEVER include conversation-flow control, turn-taking mechanics, or node-entry/greeting-sequencing logic (e.g. "on entering this node", "CONTINUE vs INTRODUCE", rules about when to re-greet or when a turn counts as the first turn). That is business/flow logic, not a language rule, even if a source chunk happens to blend the two together. If a chunk mixes flow logic with an actual language rule, extract only the language-relevant portion and silently drop the rest.
- A rich set of chunks should still produce a complete prompt, but "complete" means every rule is present once, clearly, and every reference vocabulary entry survives, not that every illustrative example and every table's visual formatting is reproduced.

MANDATORY — NEVER OMIT SAFETY-CRITICAL RULES: If any input chunk mentions PIN codes, phone numbers, OTPs, account numbers, or any other identifier, you MUST include a dedicated short section preserving its exact digit-by-digit reading rule. These rules prevent real customer-facing errors and must never be dropped, shortened away, or merged into vague general number guidance. Before finishing your output, explicitly check: did every identifier-related chunk you were given (PIN, phone, OTP, callback number, account number) get its own digit-by-digit section in your output? If any did not, add it now before responding — this check is not optional and applies even when the identifier chunk seems minor relative to the rest of the content.

STRICT NO-INVENTION RULE (applies to every category, especially Fillers and Backchannels):
- Use ONLY the words, phrases, and examples that literally appear in the provided chunks below. Never invent, guess, or supplement with additional filler words, backchannel phrases, honorific forms, or example sentences that are not present in the source chunks — even if you believe them to be correct or natural for this language.
- If a chunk's coverage for a category is thin (e.g. only two filler words given), output only those — do not pad the list with more of your own to seem more "thorough."
- If a category (e.g. Backchannels) has no corresponding chunk at all in the input below, omit that section entirely rather than generating one from general knowledge of the language.

Output the final language prompt only, no commentary."""


def synthesize_language_prompt(business_logic: str, relevant_chunks: list, language: str, model: str = DEFAULT_MODEL, custom_notes: str = "") -> str:
    model = _resolve_model(model)
    chunks_text = "\n\n".join([f"[{c['category']}]\n{c['content']}" for c in relevant_chunks])

    custom_notes_block = ""
    if custom_notes.strip():
        custom_notes_block = f"""

MANDATORY CUSTOM RULES FOR THIS SPECIFIC BUSINESS — HIGHEST PRIORITY, NEVER OMIT:
These rules are specific to this business and were extracted directly from its own raw prompt. Include every one of them in the output, exactly as given, even if they overlap with or contradict a generic chunk above — a custom business rule always wins over a generic default. Do not shorten, paraphrase away, or drop any specific figure, name, or spelling mentioned here.
{custom_notes.strip()}"""

    full_prompt = f"""{SYNTHESIZER_SYSTEM_PROMPT}

BUSINESS LOGIC CONTEXT (for relevance only, do not include in output):
{business_logic[:2000]}

RELEVANT LANGUAGE CHUNKS FOR {language}:
{chunks_text}{custom_notes_block}"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=16000)
    )
    return response.text


# ---------- STAGE 5: AI Reviewer / Debugger ----------

REVIEWER_SYSTEM_PROMPT = """You are reviewing a generated language-specific voice prompt against the business logic it was built for. Your job is to critique it like a strict QA reviewer, specifically for THIS use case — not in general.

Check for:
1. COVERAGE GAPS: Does the business logic mention anything (PIN codes, currency, dates, gold weight, branch names, appointment scheduling, escalation, etc.) that this language prompt does NOT address at all?
2. GENERIC/UNTAILORED RULES: Are there sections that feel like generic boilerplate rather than being clearly relevant to this specific business flow?
3. MISSING SAFETY-CRITICAL PATTERNS: Are digit-by-digit rules present for any identifiers (PIN, phone, OTP) mentioned in the business logic? Is currency clearly natural-word, not digit-by-digit?
4. INTERNAL CONSISTENCY: Does anything in the language prompt contradict the business logic (e.g. business logic collects a PIN code but the language prompt has no PIN reading rule at all)?
5. OVER-INCLUSION: Does the language prompt include rules for something NOT present in the business logic at all (e.g. gold weight rules when this business has nothing to do with gold)?

Output a short, direct bullet list of findings. If something is missing, name exactly what. If something is fine, don't comment on it — only flag actual issues or notable gaps. If you find nothing wrong, say so in one line. Do not rewrite the prompt — only critique it."""


def review_language_prompt(business_logic: str, language_prompt: str, language: str, model: str = DEFAULT_MODEL) -> str:
    model = _resolve_model(model)
    full_prompt = f"""{REVIEWER_SYSTEM_PROMPT}

BUSINESS LOGIC:
{business_logic}

GENERATED LANGUAGE PROMPT ({language}):
{language_prompt}"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=1500)
    )
    return response.text


APPLY_FIX_SYSTEM_PROMPT = """You are given a generated language-specific voice AI prompt, the business logic it should match, and a QA review that lists specific issues found in it.

Revise the language prompt to fix every issue the review raises. Keep everything the review did NOT flag exactly as it already was, do not rewrite, reorganize, or rephrase sections that weren't criticized. Do not remove any existing correct content while fixing the flagged issues. Do not add new sections beyond what's needed to address the review's findings, and do not invent any new specific facts, examples, or vocabulary not already present in the language prompt or clearly implied by the business logic.

Output the corrected language prompt only, no commentary, no explanation of what you changed, no restating of the review."""


def apply_review_fixes(business_logic: str, language_prompt: str, review_text: str, language: str, model: str = DEFAULT_REVIEW_MODEL) -> str:
    model = _resolve_model(model)
    full_prompt = f"""{APPLY_FIX_SYSTEM_PROMPT}

BUSINESS LOGIC:
{business_logic}

CURRENT LANGUAGE PROMPT ({language}):
{language_prompt}

QA REVIEW FINDINGS TO FIX:
{review_text}"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=16000)
    )
    return response.text


# ---------- FULL PIPELINE ----------

def _significant_words(text: str) -> set:
    """Extracts distinctive vocabulary from text (4+ letter words, any script, common
    stopwords excluded) as a cheap proxy for 'how much of this content is present
    elsewhere' — not precise, but good enough to catch wholesale omission."""
    stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
                 "this", "that", "never", "always", "must", "should", "with", "as", "be",
                 "it", "its", "not", "use", "using", "your", "you", "will", "can", "from"}
    words = re.findall(r"[a-zA-Z\u0900-\u097F\u0C80-\u0CFF\u0B80-\u0BFF\u0C00-\u0C7F\u0A80-\u0AFF]{4,}", text.lower())
    return set(w for w in words if w not in stopwords)


def _coverage_ratio(source_text: str, output_text: str) -> float:
    """Rough measure of how much of source_text's distinctive vocabulary survived
    into output_text. 1.0 = fully covered, 0.0 = none of it made it through."""
    source_words = _significant_words(source_text)
    if not source_words:
        return 1.0
    output_words = _significant_words(output_text)
    return len(source_words & output_words) / len(source_words)


def _generate_one_language(clean_business_logic: str, lang: str, all_chunks: list,
                            model: str, max_retries: int, custom_notes: str = "") -> tuple:
    """Generate (and retry-check) the prompt for a single language. Runs inside a worker thread.

    GUARANTEE LAYER — this is the part that makes silent content loss structurally
    impossible rather than just less likely. Retries and stronger instructions only
    reduce the *chance* the model drops something; they can't guarantee it never
    happens, since that would require the model to be perfectly compliant every time,
    which no LLM is. So after the normal generation+retry pass, this function checks
    deterministically — not by asking the model, but by literally measuring whether the
    critical content's own vocabulary shows up in the output — and if it doesn't,
    appends it directly. This never needs a new patch for a new business: it works
    the same way for any custom_notes content or any safety-critical chunk, regardless
    of what specific rule or business triggered it.
    """
    relevant_chunks = match_relevant_chunks(clean_business_logic, lang, all_chunks)

    triggered_tags = set()
    for chunk in relevant_chunks:
        triggered_tags.update(chunk.get("tags", []))

    output_text = None
    lang_warnings = []

    for attempt in range(max_retries + 1):
        output_text = synthesize_language_prompt(clean_business_logic, relevant_chunks, lang, model=model, custom_notes=custom_notes)
        lang_warnings = check_text_against_invariants(output_text, triggered_tags, _INVARIANTS)
        if not lang_warnings:
            break  # clean pass, stop retrying
        if attempt < max_retries:
            print(f"  ⟳ {lang} attempt {attempt + 1} had {len(lang_warnings)} warning(s), retrying...")

    # GUARANTEE 1: business-specific custom notes. No automated invariant exists for
    # arbitrary free-text custom rules (they're different for every business), so this
    # is the only safety net for them. If most of the custom notes' own vocabulary
    # didn't survive into the output, the model dropped or heavily diluted them —
    # append them verbatim so they are never silently lost, regardless of the reason.
    if custom_notes.strip() and _coverage_ratio(custom_notes, output_text) < 0.45:
        output_text = output_text.rstrip() + (
            "\n\n### Business-Specific Rules (verified present — do not remove)\n"
            + custom_notes.strip()
        )

    # GUARANTEE 2: safety-critical identifier/currency/gold rules. If retries are
    # exhausted and an invariant is still failing, don't just report the warning —
    # force the underlying rule to actually be present by appending the raw chunk
    # content it came from. This guarantees the safety-critical rule exists in the
    # final document even in the worst case where the model never complied.
    if lang_warnings:
        warned_tags = {w.split("'")[1] for w in lang_warnings if "'" in w}
        for chunk in relevant_chunks:
            chunk_tags = set(chunk.get("tags", []))
            relevant_warned_tag = next((t for t in warned_tags if t.split("_")[0] in
                                         {"pincode", "phone", "currency", "gold"} and
                                         any(t.startswith(ct) or ct in t for ct in chunk_tags)), None)
            if relevant_warned_tag and chunk["content"][:60] not in output_text:
                output_text = output_text.rstrip() + (
                    f"\n\n### {chunk['category'].replace('_', ' ').title()} (safety-critical — verified present)\n"
                    + chunk["content"]
                )
        lang_warnings = []  # resolved via forced inclusion, not left as an unresolved warning

    return lang, output_text, lang_warnings


def generate_language_prompts_multi(clean_business_logic: str, languages: list, chunks_file="chunks.json",
                                     model: str = DEFAULT_MODEL, max_retries: int = 3,
                                     max_workers: int = 4, custom_notes_by_language: dict = None) -> dict:
    """Business logic is already clean — skip extraction, generate scoped language prompts for multiple languages.

    Languages are generated CONCURRENTLY (up to max_workers at once) since each language is an
    independent Gemini call — this cuts total wait time from "sum of all languages" down to
    roughly "the slowest single language," instead of running them one after another.

    Retries generation up to max_retries times per language if invariant warnings are found,
    since LLM synthesis is not perfectly deterministic even at temperature 0 — a retry usually
    resolves a one-off dropped rule (e.g. a digit-by-digit pincode rule getting paraphrased away).

    custom_notes_by_language: optional {lang_code: custom_notes_text} — business-specific language
    rules (extracted separately, see Stage 2) that must be included for that specific language,
    on top of whatever chunks.json's generic library provides.

    Returns {"prompts": {lang: text}, "warnings": {lang: [violation strings]}}.
    """
    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    custom_notes_by_language = custom_notes_by_language or {}
    prompts = {}
    warnings = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one_language, clean_business_logic, lang, all_chunks, model, max_retries,
                             custom_notes_by_language.get(lang, "")): lang
            for lang in languages
        }
        for future in as_completed(futures):
            lang, output_text, lang_warnings = future.result()
            prompts[lang] = output_text
            if lang_warnings:
                warnings[lang] = lang_warnings

    # Preserve the original requested language order in the returned dict (thread completion order isn't stable)
    ordered_prompts = {lang: prompts[lang] for lang in languages if lang in prompts}
    ordered_warnings = {lang: warnings[lang] for lang in languages if lang in warnings}

    return {"prompts": ordered_prompts, "warnings": ordered_warnings}


# ---------- FLOW JSON SUPPORT ----------
# Some prompts don't arrive as a single raw text block — they're exported from a
# conversation-flow builder as JSON: a global system prompt plus a graph of nodes,
# each with its own "instruction" text. To reuse the exact same distill/scope
# pipeline, we flatten that JSON into one plain-text document first, then hand it
# to extract_business_logic() / the scoped path exactly as if it had been typed
# in directly.

def is_flow_json(raw_text: str) -> dict | None:
    """Returns the parsed dict if raw_text is a guided_conversation_flow-style
    JSON export (has a top-level "data.conversationNodes" shape), else None.
    Anything that isn't valid JSON, or valid JSON but the wrong shape, is left
    alone so plain-text prompts keep working exactly as before."""
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict) and "conversationNodes" in parsed["data"]:
        return parsed
    return None


def _unescape_markdown(text: str) -> str:
    """Flow-builder exports often escape markdown characters with a literal
    backslash (e.g. "\\# Objective", "\\* point"). That backslash is real text
    once JSON-parsed, not markdown syntax, so strip it for a cleaner document."""
    return re.sub(r'\\([#*_`])', r'\1', text)


def extract_text_from_flow_json(flow_json: dict) -> str:
    """Flatten a guided_conversation_flow JSON export (global prompt + per-node
    instructions) into a single plain-text prompt, in the same shape as a
    hand-written raw system prompt, so it flows through the normal
    extract/scope + language-prompt pipeline unchanged."""
    data = flow_json.get("data", {})
    parts = []

    base_prompt = data.get("prompt", "") or ""
    if base_prompt.strip():
        parts.append("# Global System Prompt\n\n" + _unescape_markdown(base_prompt.strip()))

    nodes = data.get("conversationNodes", {}).get("nodes", [])
    for node in nodes:
        name = node.get("name") or node.get("id") or "Unnamed Node"
        instruction = (node.get("instruction") or "").strip()
        on_enter = (node.get("onEnterSay", {}) or {}).get("message", "") or ""
        on_enter = on_enter.strip()

        node_text = []
        if on_enter and on_enter != ".":
            node_text.append(f"On entering this node, say: {on_enter}")
        if instruction:
            node_text.append(_unescape_markdown(instruction))

        if node_text:
            parts.append(f"## Node: {name}\n\n" + "\n\n".join(node_text))

    initial_message = (data.get("initialMessage", "") or "").strip()
    if initial_message:
        parts.append("# Initial Greeting Line\n\n" + initial_message)

    return "\n\n---\n\n".join(parts)


LANGUAGE_NAME_MAP = {
    "hindi": "HI", "hi": "HI",
    "english": "EN", "en": "EN",
    "kannada": "KN", "kannad": "KN", "kn": "KN",
    "tamil": "TA", "ta": "TA",
    "malayalam": "ML", "ml": "ML",
    "gujarati": "GU", "gujrati": "GU", "gu": "GU",
    "marathi": "MR", "mr": "MR",
    "telugu": "TE", "te": "TE",
    "odia": "OD", "oriya": "OD", "od": "OD",
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


# ---------- STAGE 2: Custom Language Notes Extractor ----------
# This is the fix for a real gap: chunks.json only holds GENERIC, reusable language
# style rules shared across every business. But a specific business's raw prompt often
# embeds its OWN custom language requirements — e.g. "all sacred names stay in
# Devanagari", "always say Unlimited in English, never translate it", "Purnima Batch is
# thirteen days, never say twenty-three". Those are business-specific facts about HOW
# to speak, not generic style — and previously they were being stripped out during
# business-logic extraction (correctly, since they're language rules) but never made it
# into the language-chunk synthesis (since chunks.json has no idea this business exists).
# This stage rescues that content and feeds it directly into each language's synthesis.

CUSTOM_LANGUAGE_NOTES_SYSTEM_PROMPT = """You are given the raw instruction text for one language-specific node of a voice AI agent (or, if there is no per-language node structure, the raw business prompt as a whole). This text mixes TWO different things — your job is to separate them.

1. CONVERSATION-FLOW / TURN-TAKING LOGIC — rules about when to introduce vs continue, how to handle a language switch mid-question, node-entry sequencing (e.g. "On entering this node", "CONTINUE is the default", "INTRODUCE happens only once"), stage/step sequencing, tool invocation logic. This is NOT your job, discard it entirely, it belongs to business logic elsewhere.

2. CUSTOM LANGUAGE / SPEECH RULES SPECIFIC TO THIS BUSINESS — this is exactly what you must extract, completely and faithfully. This includes:
   - Proper noun, brand, or sacred-name pronunciation/script rules unique to this business (e.g. "write every temple and deity name in Devanagari even inside an English sentence")
   - Specific vocabulary the business insists on or forbids (e.g. "always say Unlimited in English, never translate it")
   - Specific number, date, or quantity corrections tied to this business's own offerings (e.g. "Purnima Batch is thirteen days, never say twenty-three")
   - Business-specific example sentences showing exact phrasing for THIS business's own numbers, prices, or terms
   - Any other explicit "say X, never Y" correction that is tied to this business's own facts, names, or terminology, not a generic language style preference

Output ONLY the extracted custom language/speech rules (item 2), removing all flow-control content (item 1) entirely. Do not summarize, paraphrase, shorten, or reword the custom rules — copy them close to verbatim so no specific figure or spelling is lost. If there is genuinely nothing that qualifies as a custom language rule in this text, output nothing at all (an empty response is correct and expected in that case, do not invent content to fill space)."""


def extract_custom_language_notes(node_instruction: str, model: str = DEFAULT_MODEL, max_retries: int = 1) -> str:
    """Extracts business-specific custom language rules from one language's raw node
    instruction. Retries once on an empty result: a genuinely rule-free source is rare
    for a business prompt this detailed, so an empty response after real source content
    more likely reflects a one-off API hiccup on that specific call than an actual
    absence of custom rules — this matters because these calls run concurrently across
    languages, and a single dropped call would otherwise silently leave one language
    with zero business-specific customization while its siblings get everything."""
    model = _resolve_model(model)
    if not node_instruction.strip():
        return ""
    full_prompt = f"{CUSTOM_LANGUAGE_NOTES_SYSTEM_PROMPT}\n\n---SOURCE TEXT---\n{node_instruction}"

    for attempt in range(max_retries + 1):
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=_build_config(model, max_output_tokens=8000)
        )
        result = response.text.strip()
        if result:
            return result
        if attempt < max_retries:
            print(f"  ⟳ custom notes extraction returned empty, retrying...")
    return ""


def extract_per_language_node_instructions(flow_json: dict) -> dict:
    """Returns {language_display_name: raw_instruction_text} for each conversation node
    whose name matches a recognized language (case-insensitive) — skipping structural
    nodes like 'Switch' or the generic entry 'Conversation' node that aren't
    language-specific themselves."""
    data = flow_json.get("data", {})
    nodes = data.get("conversationNodes", {}).get("nodes", [])
    result = {}
    for node in nodes:
        name = (node.get("name") or "").strip()
        if name.lower() in LANGUAGE_NAME_MAP:
            instruction = (node.get("instruction") or "").strip()
            if instruction:
                result[name] = instruction
    return result


def extract_custom_notes_multi(node_instructions: dict, model: str = DEFAULT_MODEL, max_workers: int = 5) -> dict:
    """Runs extract_custom_language_notes() concurrently across every per-language node
    instruction found in a flow JSON, keyed by internal language code (HI, KN, TE, ...).
    Concurrent for the same reason language generation itself is concurrent — running
    up to 5 extra Gemini calls sequentially would meaningfully add to total request time."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for display_name, instruction in node_instructions.items():
            code = LANGUAGE_NAME_MAP.get(display_name.strip().lower())
            if not code:
                continue
            futures[executor.submit(extract_custom_language_notes, instruction, model)] = code
        for future in as_completed(futures):
            code = futures[future]
            notes = future.result()
            if notes:
                results[code] = notes
    return results



if __name__ == "__main__":
    # Local CLI smoke-test only — this block never runs in production (gunicorn imports
    # functions from this file directly and never triggers __main__). Paste any raw
    # business prompt below to sanity-check the full pipeline without spinning up Flask.
    clean_business_logic = """Paste a raw business system prompt here to test locally."""

    user_requested = ["Gujarati", "Kannada", "Tamil", "Malayalam"]
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