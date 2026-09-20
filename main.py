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
    "gemini-3.1-pro-preview": {
        "label": "Gemini 3.1 Pro",
        "group": "Pro",
        "supports_thinking_level": True,
        # No explicit level list — trusted with whatever thinking_level it's
        # called with (this was already the behavior before per-model level
        # lists existed, and this codebase never calls it with anything other
        # than the "low" default).
    },
    "gemini-3.6-flash": {
        "label": "Gemini 3.6 Flash",
        "group": "Flash",
        "supports_thinking_level": True,
        "thinking_levels": ["MINIMAL", "LOW", "MEDIUM", "HIGH"],
    },
    "gemini-3.8-flash": {
        "label": "Gemini 3.8 Flash",
        "group": "Flash",
        "supports_thinking_level": True,
        # MINIMAL is deliberately absent — 3.8 Flash doesn't support it per
        # Google's docs, so _build_config below must never send it.
        "thinking_levels": ["LOW", "MEDIUM", "HIGH"],
    },
}

DEFAULT_MODEL = "gemini-3.1-pro-preview"
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
        level = thinking_level
        allowed_levels = SUPPORTED_MODELS[model].get("thinking_levels")
        if allowed_levels and level.upper() not in allowed_levels:
            # e.g. "minimal" isn't valid for gemini-3.8-flash — fall back to
            # "low" rather than sending a level the model will reject; "low"
            # is valid for every model in SUPPORTED_MODELS above.
            level = "low"
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)

    return types.GenerateContentConfig(**kwargs)


# ---------- STAGE 1: Business Logic Extractor ----------

EXTRACTOR_SYSTEM_PROMPT = """You are a prompt-cleaning tool. You will be given a raw AI agent system prompt that mixes business logic (conversation flow, node structure, tool calls, guardrails, knowledge base, agent identity/persona) with language-specific rules (grammar, gender forms, colloquial speech, honorifics, number/currency/date formatting, backchannels, fillers, pronunciation rules, script rules, which spoken language to use and when).

Your job: output ONLY the business logic. Remove every language-specific rule, grammar rule, gender-form rule, colloquial-speech instruction, honorific rule, pronunciation rule, formatting-of-numbers/dates/currency-in-speech rule, AND any rule about which spoken language to default to or switch to (e.g. "primary language is Kannada, switch to English if the customer speaks English," "default to Hindi unless the customer requests otherwise"). This last category is easy to miss because it can read like a persona trait rather than an obvious grammar/pronunciation rule — treat any instruction naming a spoken language and describing when to use or change it as a language-specific rule to remove, every time, regardless of how it's phrased or where in the document it appears. This system always generates single-language, non-switching output for whichever language was requested — a leftover "switch to X" instruction from the raw prompt would directly contradict that and must never survive extraction.

This includes a specific pattern that's easy to miss because it looks like ordinary tool/integration plumbing rather than a language rule: logic naming which PRE-LOCALIZED TOOL-RESPONSE FIELD to speak from, where the field name itself encodes a spoken language (e.g. "use the speak_hi field by default; use speak_en if the customer speaks English," "respond from the hi_response field unless the en_response field applies"). This is language-selection/routing logic wearing a technical disguise — remove it completely, exactly like an explicit "switch to X" instruction, regardless of how it's named or phrased.

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

HARDCODED EXACT-WORDING SCRIPT IN ONE SPECIFIC LANGUAGE — another distinct failure mode: a business prompt sometimes gives a mandatory line as a literal quoted sentence already written out in one specific spoken language (e.g. an "openingStatement" or a "say exactly" block written entirely in Hindi, with variables like {{customerName}} embedded in it). That literal sentence is NOT reusable across languages — the business logic you output here is shared by every target language's generation, and a hardcoded Hindi sentence carried through unchanged would surface verbatim inside a Kannada or Tamil output. In this case, do not keep the literal foreign-language sentence itself; instead restate the SAME instruction in English, generic and self-contained: what must be communicated, in what order, with which variables, and any part that is genuinely fixed regardless of language (a brand name that must not be translated, a toll-free number, a compliance disclosure that must stay in English even inside another language's script). Keep every variable placeholder and every sequencing/compliance requirement exactly as strict as the original ("this must be the very first line," "do not proceed until this is confirmed," etc.) — only the literal source-language sentence itself gets generalized into an instruction, never dropped or loosened.
- Example: `openingStatement: "Hello, मैं चोलामंडलम से मिकेश बोल रहा हूँ। क्या मेरी बात {{customerName}} जी से हो रही है?"` becomes something like: "Opening line (say this first, before anything else): greet the customer, introduce yourself as Mikesh from Chola, then ask to confirm you are speaking with {{customerName}}." The requirement (greet, self-introduce as Mikesh from Chola, confirm identity, use the name exactly once here) is business logic and must be kept in full; the literal Hindi wording is not.
- This does NOT apply to a business term, brand name, or acronym that is meant to stay in English/its original form inside every language's output (e.g. "always say 'Unlimited' in English, never translate it") — that is a genuine cross-language constraint and must be kept exactly as written, quotes and all.

CONVERSATIONAL PACING AND TURN-TAKING RULES ARE BUSINESS LOGIC, NOT LANGUAGE RULES — keep them in full. This includes: turn-length ceilings (e.g. "fifteen words before pausing"), "ask one question at a time," instructions about not repeating the same phrasing across consecutive turns, instructions to phrase lines differently each time, and any scope-limiting instruction about what to read aloud (e.g. "read only the branch name and address"). None of these are about HOW to pronounce something in a specific language, they are rules about conversational structure and must be preserved exactly like any other guardrail.

THIS IS A COPY-AND-STRIP OPERATION, NOT A SUMMARY. Do NOT change, reorder, shorten, paraphrase, or condense the business logic itself, only delete language-specific sentences wholesale and leave everything else byte-for-byte as written. If you are unsure whether a sentence is a language rule or business logic, default to KEEPING it. The output should be nearly the same length as the input, minus only the language-specific portions. A noticeably shorter output than the input (accounting only for removed language content) means you have summarized, which is a failure — go back and include everything you dropped.

Output the cleaned prompt only, no commentary."""


def extract_business_logic(raw_prompt: str, model: str = DEFAULT_MODEL) -> str:
    model = _resolve_model(model)
    full_prompt = f"{EXTRACTOR_SYSTEM_PROMPT}\n\n---RAW PROMPT---\n{raw_prompt}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        # Business content must not inherit the language prompt's length budget.
        config=_build_config(model)
    )
    business_logic = response.text or ""

    # Guardrail A defense-in-depth: EXTRACTOR_SYSTEM_PROMPT asks the model not to
    # let any language-switching/routing rule survive extraction (including the
    # tool-field speak_hi/speak_en pattern), but asking nicely is never this
    # codebase's only line of defense for a hard guardrail — the same deterministic
    # detector used to filter every OTHER injection point (main body, custom notes,
    # customizations) is applied here too, since a leftover routing rule in
    # business logic would otherwise get re-matched as "relevant" content and
    # risk resurfacing in custom-notes extraction downstream.
    business_logic, removed = _strip_language_switching_instructions(business_logic)
    if removed:
        print(f"  ⚠ Removed {removed} language-switching/routing sentence(s) from extracted business logic.")
    return business_logic


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
    # Split from a single "finance_terms" trigger: Aadhaar/PAN are generic identity-document
    # references that show up across many verification-heavy businesses (gold buying, real
    # estate, subscriptions) regardless of whether that business does any lending or insurance
    # at all — bundling them with EMI/loan/insurance vocabulary meant a business that only
    # needed "Aadhaar as an ID proof" got a full loan/insurance vocabulary dump it never needed.
    "identity_documents": ["aadhaar", "pan card", "pan number"],
    "lending_insurance_terms": ["loan", "emi", "kyc", "insurance", "mutual fund", "sip", "portfolio",
                       "cibil", "credit score", "interest rate",
                       "insurance premium", "insurance policy", "nbfc", "gold loan",
                       "personal loan", "home loan"],
}

ALWAYS_ON_TAGS = ["colloquial", "honorifics", "agent_gender", "call_opening", "call_closing",
                   "backchannels", "fillers", "numbers_general", "escalation", "sensitive_situation",
                   "hold_pause", "interruption", "preserve_english_terms", "language_commitment",
                   "no_echo", "gender_neutral_guarantee"]


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

LANGUAGE_OUTPUT_GUIDANCE = """OUTPUT SCOPE AND LENGTH: The roughly 3,000-3,500 token budget applies ONLY to the language prompt, never to the business prompt or its facts. Do not pad sparse source material to reach that target.

BUSINESS ALIGNMENT: The supplied business/system prompt governs business facts, permissions, constraints, and conversation flow. This language layer governs HOW to express them, never changes WHAT the agent may do. Use business context to select relevant speech rules, not to copy the business prompt. Do not invent validation requirements, promises, escalation actions, product facts, or agent identity. Business-specific language requirements override generic language defaults; retain the configured single-language commitment. If source requirements cannot be reconciled, the reviewer must flag the conflict rather than silently invent a resolution.

USEFUL LANGUAGE COVERAGE: Internally map the actual business tasks to the supplied language rules. Prioritize natural register, respectful address, consistent agent self-reference, clear questions, appropriate acknowledgement, and pronunciation of the terms and values this business actually uses. Include only supported, relevant guidance; do not add finance, medical, scheduling, or other domain sections simply to fill a template. For a rule where an example materially prevents an error, choose one relevant example from the supplied sources. Never invent translated vocabulary or example sentences to make the prompt seem richer. Explain when a backchannel or filler fits, rather than requiring it mechanically regardless of context. Honor explicit source frequency limits.

FINAL EDITORIAL CHECK: Each distinct instruction appears once. Merge overlapping sections and omit explanatory translations from spoken vocabulary lists. Examples must use the configured agent gender, respectful customer address, and permitted register. Remove irrelevant examples rather than changing business facts to fit them. Do not claim the output is perfect, verified, or production-ready.

COMPACT DATES AND TIMES — ALL TARGET LANGUAGES: Include date/time guidance only when relevant to the business and supported by the supplied language rules. Combine it into one short paragraph (normally 2-4 sentences), with at most ONE source-provided example across dates and times together. State the pronunciation pattern and meaningful exceptions; never reproduce minute-by-minute mappings, year lists, calendar examples, or time lookup tables. This overrides instructions to reproduce reference entries for dates and times. Preserve exact date/time values, required format, day-period distinctions, and any explicit business-specific pronunciation correction. Do not invent missing language rules or silently change a supplied time. Omit the section if it is irrelevant.

COMPACT NUMBERS AND CURRENCY - ALL TARGET LANGUAGES: Use one short paragraph, normally 2-4 sentences, with at most one source-provided example. Omit generic digit, tens, hundreds, and large-number lookup lists; describe the number system instead. Preserve meaningful exceptions, zero pronunciation, currency-unit placement, and distinctions between quantities and identifiers. Apply currency decimal rules only to money, not to every decimal quantity. Include currency guidance only when relevant to the business. Explicit business-specific pronunciation corrections must survive. This overrides any requirement to reproduce generic number-word reference entries.

CONSISTENCY CHECK: Examples must obey their accompanying rules. Never combine incompatible year-reading instructions. Quantity rules must explicitly exempt identifiers, which retain their digit-by-digit or character-by-character rules. Keep business identity, factual definitions, validation lengths, and flow control in business logic; include only their language-specific pronunciation or phrasing requirements here.

Spend the remaining language budget on useful, source-backed guidance: natural phrasing, respectful address, agent gender, pronunciation of business terms, identifier and amount reading, and relevant conversational language. Keep distinct rules and corrections, merge repetition, and add no unsupported vocabulary, facts, or examples."""

SYNTHESIZER_SYSTEM_PROMPT = """You are given a business-logic system prompt for a voice AI agent, and a set of relevant detailed language-specific rule chunks for a target language.

Merge the chunks into ONE well-organized, natural-reading language prompt — the kind a senior prompt engineer would hand-write and ship to production, not an exhaustive training manual.

WRITE LIKE A HUMAN EXPERT, NOT A SPEC DOCUMENT:
- Every rule must survive, but express it in tight, flowing prose for general rules — a short paragraph per rule, not a bullet-catalog of every possible case.
- NEVER use markdown pipe-tables ( | col | col | ). That specific syntax is banned.
- TWO DIFFERENT THINGS, DO NOT CONFUSE THEM:
  1. ILLUSTRATIVE EXAMPLES (✓/✗ pairs, sample sentences showing how a grammar rule behaves): at this document's length budget, keep exactly ONE — the single clearest one — per rule, in prose. A second or third example is not more helpful, it is pure length with no new information, and this budget cannot afford it.
  2. ESSENTIAL REFERENCE DATA WITH MANY DISCRETE ENTRIES (excluding generic number, date, and time lookup lists covered by the compact-output rules below) — reproduce this COMPLETELY, every entry, never trim, sample, or abbreviate with "etc." or similar. But use the right format for what kind of data it actually is:
     - MAPPING TABLES (required acronym pronunciations, digit-by-digit letter readings — anything genuinely two-column, an input mapped to an output): use a clean bulleted list, one entry per line, NOT a pipe-table. A line like "7:15 -> ಏಳಕ್ಕೆ ಕಾಲು" is far more scannable than the same content buried in a sentence, so this format earns its space.
     - FLAT VOCABULARY LISTS (a set of preserved English terms, a backchannel word list, a filler list — no mapping, just individual words or short phrases): write these as a single compact comma-separated line, not one bullet per word. There's nothing to "scan" in a one-word bullet the way there is in a mapping, so the bullet formatting there is pure overhead with no readability benefit — every term still survives, just without a wasted line per word.
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

LANGUAGE-NAME CONSISTENCY — CHECK THIS EXPLICITLY, two distinct cases:

CASE 1 — SUBSTITUTION: business logic sometimes contains a leftover reference to whichever language it was originally written for (e.g. "numbers are spoken in Kannada words" even though you are generating for Telugu right now). If a specific spoken language is named for HOW numbers, currency, dates, or identifiers should be pronounced, and that named language is NOT the one you are generating for, correct it silently to the actual target language — never carry a mismatched language name into output meant to describe your own pronunciation.

CASE 2 — OMISSION: business logic or custom notes sometimes contain a rule that is only meaningful when the agent is speaking a SPECIFIC OTHER language, and there is no sensible substitution for it in your language (e.g. "accept English road names within Kannada or Hindi sentences" only makes sense in a Kannada or Hindi document, and rewriting it as "within English sentences" for an English document is nonsensical, English sentences don't need special permission to contain English road names). Also watch for a rule scoped to a specific language's grammar with no equivalent in yours (e.g. a note about feminine Hindi verb conjugation appearing in an English-language document, where English has no grammatical gender to apply it to). In every such case, drop the rule entirely from documents where it doesn't apply, rather than translating it or force-fitting it in — it was written for a different language's output, not yours.

Before finishing, scan your own output for the name of any language other than the one you are generating right now. For each one you find: if it's Case 1 (a pronunciation instruction with a fixable name), correct the name. If it's Case 2 (a rule that only makes sense for that other language), delete the sentence entirely. Do not leave a rule in that only makes grammatical or logical sense in a different language than the one you are writing.

NEVER INSTRUCT SWITCHING TO A DIFFERENT SPOKEN LANGUAGE, EVER — this system generates a single, non-switching language prompt per request, and a hard rule guaranteeing that (single language only, never switch, never claim inability to speak it) is always prepended to your output separately. The business logic context below is sometimes the ORIGINAL raw prompt for a business that supported multiple languages with its own switching logic (e.g. "primary language is Kannada, switch to English if the customer speaks English") — that switching behavior is exactly what this system replaces, one clean document per language. Do not reproduce it, reference it, or write your own version of it (e.g. "default to Kannada; switch to English if the customer is clearly speaking English") anywhere in your output, even inside a "Persona" or "Core Style" section where it might otherwise sound like natural framing. Describing the default/primary language itself is fine and often useful (e.g. "you speak Kannada"); describing any condition under which you would switch away from it is never allowed, in any section, regardless of what the business logic implies.

LENGTH BUDGET — HARD, NOT A SUGGESTION: the finished document must land at roughly 3,000 to 3,500 tokens total. This is a large cut from what this kind of document used to run (8,000-10,000 tokens), and hitting it requires you to actually cut, not just to write a little tighter than before. Treat every sentence as something you have to justify keeping, not something you keep by default:
- One example per rule, not two or three. This is the single biggest lever you have — a rule with three illustrative examples and the same rule with one clear example teach the model the same pattern; the extra two are pure length with zero added correctness.
- Preserve required vocabulary, acronym readings, and business-specific corrections. Generic number, date, and time lookup lists are not mandatory reference data: summarize their patterns and meaningful exceptions using the compact-output rules below. Do not derive additional entries or pad the output with examples.
- Write every general rule as the shortest sentence that still states it correctly and completely. If a rule can be said in one sentence, do not spend two on it.
- If, even after cutting every extra example and every non-essential sentence, the full set of matched chunks for this business and language genuinely cannot fit a complete and correct document into this budget, do not start dropping whole rules or categories to hit the number — a shorter version of every distinct rule is correct; a complete version of only some rules is not. In that situation, go over budget rather than silently omit a rule, and compress everything as hard as you can first.

Output the final language prompt only, no commentary.""" + "\n\n" + LANGUAGE_OUTPUT_GUIDANCE


def synthesize_language_prompt(business_logic: str, relevant_chunks: list, language: str, model: str = DEFAULT_MODEL,
                               custom_notes: str = "", custom_instruction: str = "",
                               customization_category: str = None, agent_gender_directive: str = "") -> str:
    model = _resolve_model(model)
    chunks_text = "\n\n".join([f"[{c['category']}]\n{c['content']}" for c in relevant_chunks])

    # Centrally-determined agent gender (see detect_agent_gender / generate_language_prompts_multi)
    # — placed prominently, ahead of the relevant chunks, so the model resolves the
    # agent_gender chunk's own conditional grammar guidance against ONE shared fact
    # instead of inventing or defaulting a gender independently per language. This is
    # the actual fix for two languages asserting OPPOSITE genders for the same
    # unstated persona in the same request.
    agent_gender_block = f"\n\n{agent_gender_directive.strip()}" if agent_gender_directive.strip() else ""

    # A per-request "Customise Prompt" instruction is now GUARANTEED to appear as
    # its own labeled section in the final output, deterministically appended by
    # code after this function returns (see _generate_one_language) — never left to
    # the model's own compliance. That deterministic append is what actually fixes
    # the reliability problem (it worked for 2 of 3 languages in the same request
    # under the old "ask the model to merge or create a section" design); this
    # block's only remaining job is to let the customization's ACTUAL EFFECT still
    # show up naturally within the relevant existing section's prose where useful
    # (e.g. an English-currency customization should make the Currency section's
    # own examples actually use English), without the model also trying to render
    # its own redundant section for it.
    customization_block = ""
    if custom_instruction.strip():
        category_note = (
            f" This relates to your existing '{customization_category.replace('_', ' ')}' guidance above — "
            f"where relevant, apply its actual effect within that section's own prose (e.g. change how "
            f"an example in that section is written), rather than only restating the rule."
            if customization_category else ""
        )
        customization_block = f"""

BUSINESS-SPECIFIC CUSTOMIZATION FOR THIS GENERATION — apply its real effect wherever relevant in your document:
{custom_instruction.strip()}{category_note}
A separate, clearly labeled section for this exact instruction is appended automatically after your output — do not create a section titled "Custom Instruction" (or similar) yourself, and do not mention that this is a customization, override, or special instruction; just let it naturally affect the relevant part of what you write."""

    custom_notes_block = ""
    if custom_notes.strip():
        custom_notes_block = f"""

MANDATORY CUSTOM RULES FOR THIS SPECIFIC BUSINESS — HIGHEST PRIORITY, NEVER OMIT:
These rules are specific to this business and were extracted directly from its own raw prompt. Include every one of them in the output, exactly as given, even if they overlap with or contradict a generic chunk above — a custom business rule always wins over a generic default. Do not shorten, paraphrase away, or drop any specific figure, name, or spelling mentioned here.
{custom_notes.strip()}"""

    full_prompt = f"""{SYNTHESIZER_SYSTEM_PROMPT}

BUSINESS LOGIC CONTEXT (for relevance only, do not include in output):
{business_logic}{agent_gender_block}

RELEVANT LANGUAGE CHUNKS FOR {language}:
{chunks_text}{customization_block}{custom_notes_block}"""

    # Every downstream guarantee layer (custom notes, invariants, script check, review)
    # operates on whatever text this function returns — none of them ever checked
    # whether that text was actually complete. A response can hit its own token cap
    # mid-generation and still come back as a normal-looking string with no error, no
    # missing-closer punctuation necessarily, nothing that "looks" cut off at a glance.
    # This checks the API's own finish_reason directly, the one reliable signal for
    # this, and retries once at double the token budget if the first attempt was
    # genuinely truncated, rather than letting a silently incomplete document flow
    # through every later stage untouched.
    token_budget = 16000
    text = ""
    for attempt in range(2):
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=_build_config(model, max_output_tokens=token_budget)
        )
        text = response.text or ""
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            pass
        truncated = finish_reason is not None and any(
            marker in str(finish_reason).upper() for marker in ("MAX_TOKENS", "LENGTH")
        )
        if not truncated:
            return text
        if attempt == 0:
            token_budget *= 2
            print(f"  ⚠ Synthesis for {language} hit its token limit mid-generation "
                  f"(finish_reason={finish_reason}) — retrying once at {token_budget} tokens.")
    return text


# ---------- STAGE 5: AI Reviewer / Debugger ----------

REVIEWER_SYSTEM_PROMPT = """You are reviewing a generated language-specific voice prompt against the business logic it was built for. Your job is to critique it like a strict QA reviewer, specifically for THIS use case — not in general.

Check for:
1. COVERAGE GAPS: Does the business logic mention anything (PIN codes, currency, dates, gold weight, branch names, appointment scheduling, escalation, etc.) that this language prompt does NOT address at all?
2. GENERIC/UNTAILORED RULES: Are there sections that feel like generic boilerplate rather than being clearly relevant to this specific business flow?
3. MISSING SAFETY-CRITICAL PATTERNS: Are digit-by-digit rules present for any identifiers (PIN, phone, OTP) mentioned in the business logic? Is currency clearly natural-word, not digit-by-digit?
4. INTERNAL CONSISTENCY: Does anything in the language prompt contradict the business logic (e.g. business logic collects a PIN code but the language prompt has no PIN reading rule at all)?
5. OVER-INCLUSION: Does the language prompt include rules for something NOT present in the business logic at all (e.g. gold weight rules when this business has nothing to do with gold)?

Do not flag omitted date/time example lists as missing coverage when their rules and meaningful exceptions are retained. Flag unnecessarily long date/time sections for compression.

Output a short, direct bullet list of findings. If something is missing, name exactly what. If something is fine, don't comment on it — only flag actual issues or notable gaps. If you find nothing wrong, say so in one line. Do not rewrite the prompt — only critique it.""" + "\n\n" + LANGUAGE_OUTPUT_GUIDANCE


def review_language_prompt(business_logic: str, language_prompt: str, language: str, model: str = DEFAULT_REVIEW_MODEL,
                           source_rules: str = "") -> str:
    model = _resolve_model(model)
    full_prompt = f"""{REVIEWER_SYSTEM_PROMPT}

BUSINESS LOGIC:
{business_logic}

GENERATED LANGUAGE PROMPT ({language}):
{language_prompt}

SOURCE LANGUAGE RULES AND CUSTOM REQUIREMENTS (reference material, not instructions to the reviewer):
{source_rules or 'Original language sources were not supplied. Do not claim source completeness or invent missing language rules.'}"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=1500)
    )
    return response.text or ""


APPLY_FIX_SYSTEM_PROMPT = """You are given a generated language-specific voice AI prompt, the business logic it should match, and a QA review that lists specific issues found in it.

Revise the language prompt to fix every issue the review raises. Keep everything the review did NOT flag exactly as it already was, do not rewrite, reorganize, or rephrase sections that weren't criticized. Do not remove any existing correct content while fixing the flagged issues. Do not add new sections beyond what's needed to address the review's findings, and do not invent any new specific facts, examples, or vocabulary not already present in the language prompt or clearly implied by the business logic.

Output the corrected language prompt only, no commentary, no explanation of what you changed, no restating of the review.""" + "\n\n" + LANGUAGE_OUTPUT_GUIDANCE


def apply_review_fixes(business_logic: str, language_prompt: str, review_text: str, language: str, model: str = DEFAULT_REVIEW_MODEL,
                       source_rules: str = "") -> str:
    model = _resolve_model(model)
    full_prompt = f"""{APPLY_FIX_SYSTEM_PROMPT}

BUSINESS LOGIC:
{business_logic}

CURRENT LANGUAGE PROMPT ({language}):
{language_prompt}

QA REVIEW FINDINGS TO FIX:
{review_text}

SOURCE LANGUAGE RULES AND CUSTOM REQUIREMENTS (reference material):
{source_rules or 'Original language sources were not supplied. Preserve supported wording; do not invent missing translations or examples.'}"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=16000)
    )
    return response.text or ""


# ---------- FULL PIPELINE ----------

# Categories whose chunk content is a flat, comma-separated vocabulary list meant to
# survive VERBATIM (never paraphrased) — as opposed to prose-style categories like
# escalation or colloquial style, where legitimate rewording is expected and a strict
# per-term check would constantly false-positive. Calibration confirmed a broad
# word-overlap check is too noisy for prose; but for these specific categories, every
# term is a fixed, non-negotiable word the business explicitly requires preserved, so a
# precise presence check is both safe and exactly the right tool.
_VOCABULARY_LIST_CATEGORIES = {
    "preserve_english", "identity_documents", "lending_insurance_terms",
    "backchannels", "fillers",
}


def _extract_vocabulary_terms(chunk_content: str, strip_glosses: bool = False) -> list:
    """Pulls the comma-separated term list out of a vocabulary-list chunk. These chunks
    follow the pattern '...instruction sentence: term1, term2, term3...' — this grabs the
    segment right after each colon, up to the next sentence boundary, and splits it on
    commas. Filters out pronunciation-mapping fragments (e.g. 'KYC = K-Y-C') and anything
    too long to plausibly be a single term, since those are prose, not list items."""
    if strip_glosses:
        # Explanatory translations are not additional spoken vocabulary.
        chunk_content = re.sub(r'\s*\([^()]*\)', '', chunk_content)
    segments = re.split(r':\s*', chunk_content)
    terms = []
    for seg in segments[1:]:
        first_sentence = re.split(r'[.\n]', seg)[0]
        for part in first_sentence.split(','):
            term = part.strip().rstrip('.').strip()
            if term and '=' not in term and len(term.split()) <= 3 and len(term) < 30:
                terms.append(term)
    return list(dict.fromkeys(terms))  # dedupe, preserve order


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


def _split_into_checkable_units(text: str) -> list:
    """Splits custom notes into independently-checkable units (one per line/bullet, or
    per sentence within a line) instead of leaving them as one aggregate blob. This is
    what lets a small, distinct rule be checked on its own merits rather than having its
    survival averaged against a large surrounding block of reference data that happens
    to survive easily — a large intact block should never be able to mask a small
    dropped one just by diluting the aggregate score."""
    lines = [l.strip().lstrip('*-•').strip() for l in text.split('\n') if l.strip()]
    units = []
    for line in lines:
        sentences = re.split(r'(?<=[.!?])\s+', line)
        units.extend(s.strip() for s in sentences if s.strip())
    return [u for u in units if len(u) > 15]  # skip trivial fragments (stray punctuation, headers)


def _adaptive_limits(business_logic_len: int, requested_max_retries: int) -> tuple:
    """Scales down the per-language workload automatically for very large inputs.

    Languages already run in parallel (see max_workers below), so requesting 3
    languages instead of 1 doesn't stack their time on top of each other. What DOES
    stack is the chain of sequential calls WITHIN one language: up to
    (max_retries + 1) synthesis attempts, then a review call, then a possible fix call
    — each one processing the full business logic as context, so each one gets slower
    as that context grows. For a huge input, 6 sequential large-context calls is enough
    to exceed a server timeout on its own, regardless of how many languages are
    requested. This trims that chain automatically rather than failing outright:
    fewer retries (the deterministic guarantee layers below still catch what a retry
    would have fixed) and skipping the extra semantic review pass for this run, since
    completing reliably matters more than the extra scrutiny when the input is this
    large. Returns (effective_max_retries, skip_review).
    """
    if business_logic_len > 80_000:
        return 1, True
    if business_logic_len > 40_000:
        return 2, False
    return requested_max_retries, False


# Unicode block for each language's own script. Only languages with a distinct,
# non-Latin script are listed — English has no script of its own to check against,
# and checking it here would just produce noise.
_SCRIPT_RANGES = {
    "HI": (0x0900, 0x097F),  # Devanagari
    "MR": (0x0900, 0x097F),  # Devanagari (shared with Hindi)
    "TE": (0x0C00, 0x0C7F),  # Telugu
    "KN": (0x0C80, 0x0CFF),  # Kannada
    "TA": (0x0B80, 0x0BFF),  # Tamil
    "ML": (0x0D00, 0x0D7F),  # Malayalam
    "GU": (0x0A80, 0x0AFF),  # Gujarati
    "OD": (0x0B00, 0x0B7F),  # Odia
    "BN": (0x0980, 0x09FF),  # Bengali
}


def _detect_script_mismatch(output_text: str, expected_lang: str, min_total: int = 50):
    """Detects wholesale wrong-language generation — not a stray word or a mismatched
    pronunciation rule (those are caught elsewhere), but an entire document written in
    a different script than the one requested. This happened in production: a request
    for Telugu came back almost entirely in Devanagari (Hindi) script, structurally
    coherent, individually correct-looking sentences, just the wrong language start to
    finish. None of the other guarantees catch this, because they all reason about
    MEANING (did this rule survive, does this vocabulary belong), and a wrong-language
    document can still look internally consistent by that measure. Script identity is
    different: it's a hard, unambiguous fact about the text, not a judgment call, so
    checking it directly catches this specific failure class with certainty rather than
    probability. Returns the wrongly-dominant language code if a mismatch is found, or
    None if the script is correct (or the language has no dedicated script to check,
    e.g. English).

    min_total lets a caller lower the "enough script content to judge reliably" floor
    below the 50-character default used for whole-document checks — a per-sentence or
    per-unit check (see _filter_custom_notes_for_wrong_language) needs a much smaller
    floor, since a single contaminated sentence will rarely reach 50 characters of
    script on its own, and a fixed document-sized threshold there would let short
    contaminated fragments through undetected."""
    if expected_lang not in _SCRIPT_RANGES:
        return None
    counts = {lang: 0 for lang in _SCRIPT_RANGES}
    for ch in output_text:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
                break
    total = sum(counts.values())
    if total < min_total:  # not enough script content in the output to judge reliably either way
        return None
    expected_share = counts[expected_lang] / total
    if expected_share >= 0.3:  # expected script is clearly present, not a wholesale mismatch
        return None
    wrong_lang = max((l for l in counts if l != expected_lang), key=lambda l: counts[l])
    if counts[wrong_lang] > counts[expected_lang]:
        return wrong_lang
    return None


# Hindi and Marathi are the one pair in this system that share a script — both
# Devanagari — so _detect_script_mismatch's Unicode-range check structurally cannot
# tell them apart; text that's correctly identified as "Devanagari" can still be the
# wrong LANGUAGE within that script. This happened in production: custom notes
# extracted from a business prompt's Hindi section ("You only speak in Hindi, always")
# got fed into a Marathi generation, rendered in perfectly valid Devanagari, and passed
# the script check cleanly since it never checked language identity, only script.
# These marker words are extremely common, high-frequency, and genuinely unambiguous
# between the two languages — not a stylistic difference, but entirely different words
# for the same basic grammatical function (existential "is/are", negation, self-reference).
_HI_MR_MARKERS = {
    "HI": ["है", "हैं", "नहीं", "मुझे", "करता हूं", "करती हूं"],
    "MR": ["आहे", "आहेत", "नाही", "मला", "करतो", "करते"],
}


def _detect_hindi_marathi_confusion(text: str, expected_lang: str, min_markers: int = 5):
    """Only runs when expected_lang is HI or MR, since that's the only same-script pair
    in this system. Returns the other language's code if its markers clearly dominate,
    or None if the expected language's own markers are present as they should be.

    min_markers lets a caller lower the "enough markers to judge reliably" floor below
    the whole-document default of 5 — see _detect_script_mismatch's min_total for why a
    per-unit check needs a smaller floor than a whole-document one."""
    if expected_lang not in ("HI", "MR"):
        return None
    other_lang = "MR" if expected_lang == "HI" else "HI"
    expected_count = sum(text.count(m) for m in _HI_MR_MARKERS[expected_lang])
    other_count = sum(text.count(m) for m in _HI_MR_MARKERS[other_lang])
    if expected_count + other_count < min_markers:  # too few markers present to judge reliably
        return None
    if other_count > expected_count * 1.5:
        return other_lang
    return None


def _filter_custom_notes_for_wrong_language(custom_notes: str, lang: str) -> tuple:
    """Checks each individually-checkable unit of custom_notes for wrong-language
    contamination, rather than treating the notes as one aggregate blob.

    This closes a real gap found in production: a Kannada generation's custom notes
    were mostly genuine Kannada-language business rules, but also carried one short
    leftover sentence quoted verbatim from the business's Hindi section. Checking the
    notes in aggregate (as this used to do) counts script characters across the WHOLE
    blob before judging — a large block of genuine Kannada script easily pushed the
    Kannada share back above the "clearly present" threshold, hiding the short Hindi
    fragment inside it. That's the exact same dilution problem already solved for
    Guarantee 1's coverage check (a large intact block masking a small dropped one) —
    it just wasn't ever applied to this wrong-language filter. Checking each unit
    (roughly one sentence) on its own, with a threshold sized for a single sentence
    rather than a whole document, catches what the aggregate check structurally
    cannot: a large correct block can no longer hide a small contaminated one.

    Returns (filtered_notes, discarded_unit_count).
    """
    if not custom_notes.strip():
        return custom_notes, 0
    units = _split_into_checkable_units(custom_notes)
    if not units:
        return custom_notes, 0
    kept, discarded = [], 0
    for unit in units:
        wrong = (_detect_script_mismatch(unit, lang, min_total=15)
                 or _detect_hindi_marathi_confusion(unit, lang, min_markers=2))
        if wrong:
            discarded += 1
        else:
            kept.append(unit)
    return "\n".join(kept), discarded


# ---------- GUARDRAIL B: language-switching/imposition detector ----------
# One canonical detector, used at every point ANY text is about to be injected into
# a generation — the main synthesized body, custom notes extracted from a raw
# business prompt, and a user-typed customization. Cross-language contamination has
# been found coming from all three sources; the earlier fix only ever ran on the
# main body, once, at the end. This is deliberately broader than literal "switch to
# X" phrasing: it also catches a default-language declaration with a conditional
# fallback to another language, an exclusive-single-language assertion regardless of
# which language it names, and this domain's tool-field routing convention
# (speak_hi/speak_en and similar field-name patterns) — all of these are the same
# underlying violation (asserting or routing spoken-language identity), just phrased
# differently depending on which pipeline stage the text came from.
_LANG_NAME_ALT = (r'(?:english|hindi|kannada|tamil|malayalam|gujarati|marathi|telugu|odia|oriya|bengali)')
_LANG_FIELD_CODE_ALT = r'(?:en|hi|kn|ta|ml|gu|mr|te|od|bn)'

_LANGUAGE_IMPOSITION_PATTERNS = [
    # "switch to English", "switching into Kannada", "switch back to Hindi"
    re.compile(rf'\bswitch(?:es|ing|ed)?\s+(?:back\s+)?(?:to|into)\s+(?:speaking\s+)?{_LANG_NAME_ALT}\b', re.IGNORECASE),
    # "speak/respond/converse (only) in X" — tight adjacency on purpose (see note
    # below) so this doesn't fire on an unrelated sentence like "always speak the
    # branch name in English", which has real words between the verb and "in X".
    re.compile(rf'\b(?:speak|talk|converse|communicate|respond|reply)\w*\s+(?:only\s+|solely\s+|purely\s+|always\s+)?(?:in|using)\s+{_LANG_NAME_ALT}\b', re.IGNORECASE),
    # "you (only) speak only X" / "speak only Hindi" (no "in") / "you only speak X"
    # (only before the verb instead of after) — a business declaration of exclusive
    # single-language use, banned regardless of phrasing or which language it names
    # per Guardrail B, since this duplicates what the deterministic
    # language_commitment guarantee already owns.
    re.compile(rf'\b(?:you\s+)?(?:only\s+)?(?:speak|talk|converse|respond)\w*\s+only\s+(?:colloquial\s+)?{_LANG_NAME_ALT}\b', re.IGNORECASE),
    re.compile(rf'\b(?:you\s+)?only\s+(?:speak|talk|converse|respond)\w*\s+(?:colloquial\s+)?(?:in\s+)?{_LANG_NAME_ALT}\b', re.IGNORECASE),
    # "use Kannada language throughout"
    re.compile(rf'\buse\s+{_LANG_NAME_ALT}\s+(?:language\s+)?throughout\b', re.IGNORECASE),
    # "default/primary/preferred (spoken) language is Hindi"
    re.compile(rf'\b(?:default|primary|preferred)\s+(?:spoken\s+)?language\s+(?:is|:)\s*{_LANG_NAME_ALT}\b', re.IGNORECASE),
    # this domain's tool-field routing convention: speak_hi, speak_en, hi_response,
    # response_en, en_field, etc. — these field names exist ONLY to route between
    # spoken languages, so their mere presence in text destined for a language
    # document is itself the violation, regardless of surrounding wording.
    re.compile(rf'\bspeak[_\s]?{_LANG_FIELD_CODE_ALT}\b', re.IGNORECASE),
    re.compile(rf'\b{_LANG_FIELD_CODE_ALT}[_\s]?(?:speak|response|reply|field|flag|version|output|text)\b', re.IGNORECASE),
    # conditional fallback between two named languages: "Hindi ... if/unless/when
    # the customer speaks/prefers/asks for ... English" — the Spinny
    # speak_hi/speak_en pattern spelled out in prose instead of field names.
    re.compile(
        rf'\b{_LANG_NAME_ALT}\b[^.!?\n]{{0,80}}\b(?:if|unless|when)\b[^.!?\n]{{0,60}}'
        rf'\bcustomer\b[^.!?\n]{{0,40}}\b(?:speaks?|prefers?|asks?\s+for|wants?)\b[^.!?\n]{{0,20}}{_LANG_NAME_ALT}\b',
        re.IGNORECASE,
    ),
]


def _unit_imposes_language_switch(text: str) -> bool:
    """The single canonical check behind Guardrail B: true if this piece of text —
    regardless of which pipeline stage it came from (main synthesis output,
    extraction-sourced custom notes, or a user-typed customization) — asserts or
    routes between spoken languages in a way no generated language document may
    ever contain. Every call site in this file that needs this check (the main-body
    stripper, the custom-notes filter, and the customization rejection check) calls
    this exact function, so a fix here fixes all three at once rather than needing
    three separate patches that happen to agree with each other."""
    if not text or not text.strip():
        return False
    return any(p.search(text) for p in _LANGUAGE_IMPOSITION_PATTERNS)


# The one sanctioned exception to Guardrail B: this system's own deterministic
# language_commitment guarantee line ("You speak only colloquial Kannada
# naturally...") legitimately names the document's own target language, by design,
# and is held out of every internal filtering step for exactly that reason (see
# commitment_chunk in _generate_one_language). A full generated document
# necessarily contains this line, so the standing external scanner below needs to
# recognize it specifically — not exempt same-language mentions generically, which
# would also hide a genuine same-language violation elsewhere in the document
# (e.g. a redundant, non-guaranteed "default language is Kannada" sentence the
# synthesizer wrote on its own still belongs in NO document, guaranteed line or not).
_LANG_CODE_TO_FULL_NAME = {
    "EN": "english", "HI": "hindi", "KN": "kannada", "TA": "tamil", "ML": "malayalam",
    "GU": "gujarati", "MR": "marathi", "TE": "telugu", "OD": "odia", "BN": "bengali",
}


def _is_own_language_commitment_line(sentence: str, expected_language: str) -> bool:
    """True only for a sentence from this system's own fixed language_commitment
    guarantee template naming exactly the expected language — see the module
    comment above. The template is two sentences ("You speak only colloquial X
    naturally..." and "...cannot speak in X at all..."), both of which
    legitimately name X and must both be recognized, not just the first."""
    lang_name = _LANG_CODE_TO_FULL_NAME.get(expected_language)
    if not lang_name:
        return False
    patterns = (
        rf'you\s+speak\s+only\s+colloquial\s+{lang_name}\s+naturally',
        rf'cannot\s+speak\s+in\s+{lang_name}\s+at\s+all',
    )
    return any(re.search(p, sentence, re.IGNORECASE) for p in patterns)


def scan_document_for_language_violations(document_text: str, expected_language: str = None) -> list:
    """Standing automated check for Guardrail B: scans a fully generated document —
    from a live generation, or fed in by an external regression-test script — for
    any sentence naming another language in a switching/default/exclusivity
    context, using the exact same canonical detector (_unit_imposes_language_switch)
    that every injection point in this file is filtered through. Unlike the
    pipeline's own stripping guarantees (which remove what they find as part of
    generation), this is a pure, read-only detector meant to be run independently —
    against a finished document — so a future regression of this class of bug
    surfaces as a hard, visible failure in a test/verification run, rather than
    requiring someone to notice it in a live generation again.

    expected_language: when given (the language this document was generated for),
    exempts ONLY this document's own fixed language_commitment guarantee line from
    being flagged — the one sanctioned, by-design exception — while still flagging
    every other match, including a same-language default/exclusivity statement
    written anywhere else, since only the guarantee's own fixed line is exempt.

    Returns the list of offending sentences found (empty if none)."""
    if not document_text or not document_text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', document_text)
    violations = []
    for sentence in sentences:
        if not _unit_imposes_language_switch(sentence):
            continue
        if expected_language and _is_own_language_commitment_line(sentence, expected_language):
            continue
        violations.append(sentence.strip())
    return violations


def _strip_language_switching_instructions(output_text: str) -> tuple:
    """Deterministically removes any sentence asserting or routing between spoken
    languages (see _unit_imposes_language_switch) — this can only ever contradict
    the always-on language_commitment guarantee (single language, never switch),
    which is why this doesn't rely on the extractor or synthesizer prompts alone to
    keep it out. Both of those reduce the CHANCE it appears; this makes its absence
    certain.

    Per Guardrail B this is absolute — there is no "legitimate business override"
    exemption here (an earlier version of this function left room for one, on the
    theory that Guarantee 1's custom-notes injection might deliberately want
    different language behavior; that theory was itself the root cause of a real
    leak — see the custom-notes filtering in _generate_one_language, which now
    applies this exact same check to custom notes BEFORE they're ever injected,
    rather than exempting them).

    Returns (cleaned_text, count_removed).
    """
    sentences = re.split(r'(?<=[.!?])\s+', output_text)
    kept, removed = [], 0
    for sentence in sentences:
        if _unit_imposes_language_switch(sentence):
            removed += 1
        else:
            kept.append(sentence)
    if not removed:
        return output_text, 0
    cleaned = " ".join(kept)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)  # collapse a run of spaces left where a mid-line sentence was removed
    cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)  # drop a leading space left on the next line/paragraph start
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip(), removed


def _filter_custom_notes_for_language_imposition(custom_notes: str) -> tuple:
    """Applies the same Guardrail B check (_unit_imposes_language_switch) to
    custom notes — from EITHER source that feeds this channel: extraction from a
    raw business prompt (Stage 2), or a user-typed 'Customise Prompt' customization
    — before they are ever injected into a generation. This is the actual fix for
    cross-language contamination arriving via custom notes rather than the main
    synthesized body: a real business rule ('default spoken language is Hindi via
    speak_hi; use speak_en if the customer speaks English') survived extraction as
    legitimate-looking custom-notes content and leaked into every language's
    document, most visibly the English one, which ended up asserting its own
    default was actually Hindi.

    Checked per-unit (one sentence/bullet at a time, via _split_into_checkable_units)
    rather than as one aggregate blob, for the same dilution reason already fixed
    for wrong-script contamination: a single offending sentence must not be able to
    hide inside a larger block of genuinely fine custom notes. The safer default
    when a unit is flagged is to drop it entirely rather than try to rewrite it —
    Guardrail B is absolute, and a language-routing rule doesn't belong in the
    language layer at all (Guardrail A), so there is no per-language rewrite that
    would make it acceptable to keep.

    Returns (filtered_notes, removed_unit_count).
    """
    if not custom_notes.strip():
        return custom_notes, 0
    units = _split_into_checkable_units(custom_notes)
    if not units:
        return custom_notes, 0
    kept, removed = [], 0
    for unit in units:
        if _unit_imposes_language_switch(unit):
            removed += 1
        else:
            kept.append(unit)
    return "\n".join(kept), removed


# Fixed, literal phrases that only ever appear in this codebase's OWN internal
# customization-integration guidance (see synthesize_language_prompt's
# customization_guidance_block) — never legitimate customer-facing content. A
# production Kannada document once shipped with this kind of sentence verbatim
# because the guidance text was appended directly onto the customization content
# instead of being kept in its own never-reproduce-this paragraph (now fixed at the
# source). This is a deterministic safety net on top of that fix, not a replacement
# for it — the same "don't just ask nicely" principle already applied to every other
# guarantee in this file.
_CUSTOMIZATION_META_LEAK_PHRASES = [
    "strengthens/extends the existing",
    "apply it as an amendment to that category",
    "does not belong under any of the standard categories",
    "internal writing guidance",
    "do not explain this categorization",
]


def _strip_customization_meta_leak(output_text: str) -> tuple:
    """Deterministically removes any sentence containing one of this codebase's own
    internal customization-integration phrases, in case it leaked into the output
    despite being kept out of the 'include exactly as given' custom-notes channel.
    Returns (cleaned_text, count_removed)."""
    sentences = re.split(r'(?<=[.!?])\s+', output_text)
    lowered_phrases = _CUSTOMIZATION_META_LEAK_PHRASES
    kept, removed = [], 0
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(phrase in sentence_lower for phrase in lowered_phrases):
            removed += 1
        else:
            kept.append(sentence)
    if not removed:
        return output_text, 0
    cleaned = " ".join(kept)
    cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip(), removed


def _generate_one_language(clean_business_logic: str, lang: str, all_chunks: list,
                            model: str, max_retries: int, custom_notes: str = "",
                            skip_review: bool = False, customization_guarantee_domain: str = None,
                            custom_instruction: str = "", customization_category: str = None,
                            agent_gender_directive: str = "") -> tuple:
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

    # The "speak only colloquial <language>, never switch, never claim you can't"
    # rule must appear FIRST in every generated document, unconditionally — this
    # is a positioning guarantee, not a request the synthesizer LLM can place
    # wherever it judges best (or drop into the middle of a section, or paraphrase
    # into something softer). Held out of the chunks handed to synthesis entirely
    # (so the model never independently restates it elsewhere, which would risk a
    # duplicate or a paraphrase drifting from the exact required wording) and
    # prepended verbatim after every other step, including the review/fix pass,
    # so nothing downstream can move, reword, or remove it.
    commitment_chunk = next((c for c in relevant_chunks if c["category"] == "language_commitment"), None)

    # Same held-out/deterministic-append mechanism as commitment_chunk above, for the
    # same reason: gender-neutral customer address must not be left to whatever a given
    # language's honorifics chunk happens to say (coverage there is inconsistent across
    # languages) — it's guaranteed present, worded consistently, regardless of what the
    # synthesizer does on its own.
    gender_neutral_chunk = next((c for c in relevant_chunks if c["category"] == "gender_neutral_guarantee"), None)

    synthesis_chunks = [c for c in relevant_chunks
                        if c["category"] not in ("language_commitment", "gender_neutral_guarantee")]

    triggered_tags = set()
    for chunk in relevant_chunks:
        triggered_tags.update(chunk.get("tags", []))

    # Filter custom_notes for wrong-language contamination BEFORE it's used anywhere —
    # this is the actual source of the bug (custom notes extracted from the wrong
    # per-language section of the business prompt), and it feeds both the synthesis
    # prompt injection below AND Guarantee 1's append later. Catching it here stops it
    # from leaking in through either path, rather than only detecting it in the final
    # aggregated output, which has a real blind spot of its own: a large, correctly-
    # scripted body can dilute a small contaminated section below any aggregate
    # mismatch threshold, exactly the same dilution problem fixed earlier for Guarantee
    # 1's coverage check, just recurring here in a different check. Checking the custom
    # notes on their own, before they're mixed into anything larger, avoids that
    # entirely — there's nothing yet for a contaminated section to hide inside.
    #
    # This runs the general script-mismatch check (catches any two languages with
    # different scripts, e.g. Hindi custom notes leaking into Telugu) AND the
    # Hindi/Marathi lexical check (the one pair that shares a script, where the general
    # check alone is blind) — together they cover every language pair in this system.
    #
    # Checked per-unit (see _filter_custom_notes_for_wrong_language), not as one
    # aggregate blob — a production Kannada generation shipped with a Hindi-language
    # section because a short contaminated sentence was diluted below the aggregate
    # detection threshold by the larger genuinely-Kannada custom notes around it. The
    # aggregate check is kept as a second-line defense afterward in case the notes are
    # wrong-language wholesale (not just one leftover fragment).
    if custom_notes.strip():
        custom_notes, discarded_units = _filter_custom_notes_for_wrong_language(custom_notes, lang)
        if discarded_units:
            print(f"  ⚠ Discarded {discarded_units} custom-notes line(s) for '{lang}' — "
                  f"detected as wrong-language contamination.")

    wrong_lang_in_notes = None
    if custom_notes.strip():
        wrong_lang_in_notes = _detect_script_mismatch(custom_notes, lang) or _detect_hindi_marathi_confusion(custom_notes, lang)
    if wrong_lang_in_notes:
        print(f"  ⚠ Discarding remaining custom notes for '{lang}' — aggregate still detected as "
              f"{wrong_lang_in_notes}, not {lang}'s own custom notes.")
        custom_notes = ""

    # GUARDRAIL B, applied at the custom-notes injection point — this is the actual
    # fix for the tool-field language-routing rule (e.g. "default spoken language is
    # Hindi via speak_hi; use speak_en if the customer speaks English") that survived
    # extraction from a raw business prompt and leaked into every language's
    # document, most visibly the English one asserting its own default was Hindi.
    # This is the SAME detector (_unit_imposes_language_switch) used below on the
    # main synthesized body and in generate_language_prompts_multi's customization
    # rejection check — one mechanism, three injection points, not three patches.
    if custom_notes.strip():
        custom_notes, imposition_units_removed = _filter_custom_notes_for_language_imposition(custom_notes)
        if imposition_units_removed:
            print(f"  ⚠ Discarded {imposition_units_removed} custom-notes line(s) for '{lang}' — "
                  f"detected as a language-switching/imposition rule (Guardrail B).")

    output_text = None
    lang_warnings = []
    script_correction_note = ""

    for attempt in range(max_retries + 1):
        # If the previous attempt came back in the wrong script entirely, inject an
        # explicit, unmissable correction ahead of everything else for this retry —
        # this is not a minor style fix, it's telling the model the whole document was
        # in the wrong language and must not be again.
        retry_notes = (script_correction_note + "\n\n" + custom_notes).strip() if script_correction_note else custom_notes
        output_text = synthesize_language_prompt(clean_business_logic, synthesis_chunks, lang, model=model, custom_notes=retry_notes,
                                                  custom_instruction=custom_instruction,
                                                  customization_category=customization_category,
                                                  agent_gender_directive=agent_gender_directive)

        wrong_script_lang = _detect_script_mismatch(output_text, lang) or _detect_hindi_marathi_confusion(output_text, lang)
        lang_warnings = check_text_against_invariants(output_text, triggered_tags, _INVARIANTS)

        if wrong_script_lang:
            script_correction_note = (
                f"CRITICAL CORRECTION — read this first: your previous attempt for this "
                f"exact request was written almost entirely in {wrong_script_lang} instead "
                f"of {lang}. This is not a wording issue, the entire document must be in "
                f"{lang} from the first sentence onward. Do not repeat that mistake."
            )
            if attempt < max_retries:
                print(f"  ⚠ {lang} attempt {attempt + 1} came back in {wrong_script_lang}, forcing corrective retry...")
                continue  # this takes priority over invariant warnings — retry regardless of those
        else:
            script_correction_note = ""

        if not lang_warnings and not wrong_script_lang:
            break  # clean pass, stop retrying
        if attempt < max_retries and not wrong_script_lang:
            print(f"  ⟳ {lang} attempt {attempt + 1} had {len(lang_warnings)} warning(s), retrying...")

    # If a wrong-language mismatch survived every normal retry, a warning banner sitting
    # on top of an otherwise-unusable, wrong-language document is not an acceptable
    # shipped result — someone can miss or ignore the banner and use the content as-is.
    # One last, DISTINCTLY different corrective attempt is made before giving up: a
    # fresh from-scratch synthesis call (not a continuation of the failing attempts),
    # on the stronger review model rather than whichever model has already failed
    # max_retries+1 times in a row, since repeating the same model/prompt combination
    # that just failed repeatedly is unlikely to succeed a further identical way.
    final_wrong_lang = _detect_script_mismatch(output_text, lang) or _detect_hindi_marathi_confusion(output_text, lang)
    generation_failed = False
    if final_wrong_lang:
        print(f"  ⚠ {lang}: all {max_retries + 1} attempt(s) still mismatched ({final_wrong_lang}) — "
              f"making one final corrective attempt on the stronger review model.")
        last_resort_notes = custom_notes.strip() + "\n\n" if custom_notes.strip() else ""
        last_resort_notes += (
            f"LAST-RESORT CORRECTION — every previous attempt at this exact request came back "
            f"written mostly in {final_wrong_lang} instead of {lang}. Do not reuse, continue, or "
            f"lightly edit any previous attempt. Write this document completely from scratch, in "
            f"{lang} only, from the first word to the last. If you notice yourself about to write "
            f"a sentence in any language other than {lang}, stop and rewrite that sentence in "
            f"{lang} instead."
        )
        last_resort_text = synthesize_language_prompt(
            clean_business_logic, synthesis_chunks, lang,
            model=DEFAULT_REVIEW_MODEL, custom_notes=last_resort_notes.strip(),
            custom_instruction=custom_instruction,
            customization_category=customization_category,
            agent_gender_directive=agent_gender_directive
        )
        last_resort_wrong_lang = (_detect_script_mismatch(last_resort_text, lang)
                                   or _detect_hindi_marathi_confusion(last_resort_text, lang))
        if not last_resort_wrong_lang:
            print(f"  ✓ {lang}: last-resort corrective attempt succeeded.")
            output_text = last_resort_text
            lang_warnings = check_text_against_invariants(output_text, triggered_tags, _INVARIANTS)
            final_wrong_lang = None
        else:
            print(f"  ✗ {lang}: last-resort corrective attempt STILL mismatched "
                  f"({last_resort_wrong_lang}) — returning an error state instead of shipping "
                  f"wrong-language content.")
            generation_failed = True
            final_wrong_lang = last_resort_wrong_lang

    if generation_failed:
        # Never ship wrong-language content as if it were a normal, usable result — not
        # even with a warning attached. The rest of the guarantee pipeline (custom-notes
        # append, safety-critical force-append, vocabulary completion, review) all
        # assume they're refining a usable draft, which this isn't, so skip them and
        # return a clear, unambiguous error state instead.
        error_text = (
            f"⚠ GENERATION FAILED FOR {lang}.\n\n"
            f"Every generation attempt — including a dedicated final corrective attempt on a "
            f"different model — produced output predominantly in {final_wrong_lang} instead of "
            f"{lang}. Rather than return unusable wrong-language content, this generation was "
            f"blocked.\n\n"
            f"Please retry generation for {lang}, or review the business logic and custom notes "
            f"supplied for this language before trying again."
        )
        return lang, error_text, [
            f"Generation failed for '{lang}': output remained in {final_wrong_lang} after every "
            f"retry and a final corrective attempt. No usable prompt was produced."
        ]

    # GUARANTEE 5: never ship a language-switching/imposition instruction (Guardrail
    # B), applied here to the main synthesized body. custom_notes was already put
    # through the identical check above before this point, so Guarantee 1 below can
    # never reintroduce one — there is no "legitimate override" exemption for either
    # source, per Guardrail B being absolute.
    output_text, switch_removed = _strip_language_switching_instructions(output_text)
    if switch_removed:
        print(f"  ⚠ {lang}: removed {switch_removed} language-switching instruction(s) "
              f"that contradicted the language_commitment guarantee.")

    # Defense-in-depth net for the customization-integration-guidance leak (see
    # _strip_customization_meta_leak) — the root fix keeps this text out of the
    # synthesizer's "include exactly as given" channel entirely, but this catches it
    # deterministically in case it slips through anyway.
    output_text, meta_leak_removed = _strip_customization_meta_leak(output_text)
    if meta_leak_removed:
        print(f"  ⚠ {lang}: removed {meta_leak_removed} leaked internal customization-guidance "
              f"sentence(s) from the output.")

    # GUARANTEE 1: business-specific custom notes. No automated invariant exists for
    # arbitrary free-text custom rules (they're different for every business), so this
    # is the only safety net for them.
    #
    # Checking custom_notes as one aggregate blob has a real blind spot: a business
    # whose custom notes mix a large block of reference data (pricing, contacts, ride
    # lists — voluminous, and reliably survives synthesis) with a small distinct set of
    # hard guardrails can pass the AGGREGATE coverage check easily, purely on the
    # strength of the reference data surviving, even while every one of the guardrail
    # sentences inside that same blob got fully dropped. This is exactly the failure
    # mode found in production: 6 "never-flex" rules in, 1 survived, and the aggregate
    # score never flagged it because the surrounding reference data was intact.
    #
    # The fix: split custom notes into independent units (one per line/sentence) and
    # check EACH one's own survival, not the average across all of them. A large intact
    # block can no longer hide a small dropped one.
    if custom_notes.strip():
        units = _split_into_checkable_units(custom_notes)
        missing_units = [u for u in units if _coverage_ratio(u, output_text) < 0.35]
        if missing_units:
            output_text = output_text.rstrip() + (
                "\n\n### Business-Specific Rules — additional rules (verified present)\n"
                + "\n".join(missing_units)
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
            if not relevant_warned_tag or chunk["content"][:60] in output_text:
                continue
            # This check assumes in-language numbers/currency are always correct — but a
            # user's custom instruction for this exact generation (e.g. "keep currency in
            # English") can deliberately make that assumption false. Without this guard,
            # force-appending the original in-language chunk here would silently override
            # the user's explicit customization with the generic default it was meant to
            # replace. If the customization for this generation covers the domain this
            # warning is about, stand down instead of fighting it.
            if customization_guarantee_domain and relevant_warned_tag.startswith(customization_guarantee_domain):
                continue
            output_text = output_text.rstrip() + (
                f"\n\n### {chunk['category'].replace('_', ' ').title()} (safety-critical — verified present)\n"
                + chunk["content"]
            )
        lang_warnings = []  # resolved via forced inclusion (or deliberately stood down), not left as an unresolved warning

    # GUARANTEE 4: precise vocabulary-list completion. Unlike prose categories, a
    # vocabulary-list chunk (preserved terms, identity documents, lending/insurance
    # terms, backchannels, fillers) is a fixed set of individual words the business
    # explicitly requires — there's no legitimate "paraphrase" of a term list, each word
    # either made it in or didn't. This is precisely the pattern that caused real gaps
    # before (e.g. six lending/insurance terms present in one language's output but
    # silently absent from another, despite both being fed the identical chunk). Since
    # every listed term is meant to appear close to verbatim, checking for its literal
    # presence is safe here in a way a broad prose check never was — and any term that
    # didn't make it gets added directly, closing this exact gap for any business, any
    # language, any future vocabulary-list chunk, not just the one that surfaced it.
    for chunk in relevant_chunks:
        if chunk.get("category") not in _VOCABULARY_LIST_CATEGORIES:
            continue
        terms = _extract_vocabulary_terms(
            chunk["content"], strip_glosses=chunk.get("category") in {"backchannels", "fillers"}
        )
        missing_terms = [t for t in terms if t.lower() not in output_text.lower()]
        if missing_terms:
            output_text = output_text.rstrip() + (
                f"\n\n### {chunk['category'].replace('_', ' ').title()} — additional terms (verified present)\n"
                + ", ".join(missing_terms)
            )

    # GUARANTEE 3: automatic semantic review and correction, run on every generation,
    # not as an opt-in button. Word-overlap checks (guarantees 1 and 2) are fast and
    # deterministic but blind to meaning — they can't tell "escalation was paraphrased
    # well" from "escalation is missing", and testing confirmed those cases score too
    # similarly to threshold safely. A model that actually reads the output for meaning
    # can tell the difference, and catches classes of problem no keyword check ever
    # could: a wrong language mentioned where it shouldn't be, a whole category quietly
    # dropped, vocabulary that doesn't belong for this business. This runs with the
    # stronger review model regardless of which model generated the draft, since
    # catching subtle problems matters more here than matching the generation model.
    source_rules = "\n\n".join(f"[{c['category']}]\n{c['content']}" for c in synthesis_chunks)
    if custom_notes.strip():
        source_rules += "\n\n[Business-specific language requirements]\n" + custom_notes
    if custom_instruction.strip():
        source_rules += "\n\n[User customization for this generation]\n" + custom_instruction

    # Snapshot of the body BEFORE the review/fix pass — it's already been verified
    # language-correct (either it passed the retry loop cleanly, or the last-resort
    # corrective attempt above fixed it). If the review/fix pass regresses it back into
    # the wrong language, this is what gets restored, rather than shipping the regressed
    # version with just a warning attached.
    pre_review_text = output_text

    review_text = "" if skip_review else review_language_prompt(
        clean_business_logic, output_text, lang, model=DEFAULT_REVIEW_MODEL, source_rules=source_rules
    )
    if not skip_review and _review_found_issues(review_text):
        revised = apply_review_fixes(
            clean_business_logic, output_text, review_text, lang,
            model=DEFAULT_REVIEW_MODEL, source_rules=source_rules
        )
        if revised.strip():
            output_text = revised
        else:
            lang_warnings.append("Review fixes returned no text; retained the generated prompt.")

    # Corrections can introduce new regressions; check the final body, not just the draft.
    output_text, _ = _strip_language_switching_instructions(output_text)
    output_text, _ = _strip_customization_meta_leak(output_text)
    lang_warnings.extend(check_text_against_invariants(output_text, triggered_tags, _INVARIANTS))
    final_wrong_lang = _detect_script_mismatch(output_text, lang) or _detect_hindi_marathi_confusion(output_text, lang)
    if final_wrong_lang:
        # The review/fix pass (which has no explicit awareness of script/language
        # correctness — see REVIEWER_SYSTEM_PROMPT) introduced a language regression into
        # an output that was already verified correct. Never ship that regression with
        # just a warning on top — revert to the pre-review version, which is known-good,
        # and say so, rather than presenting wrong-language content as a normal result.
        print(f"  ⚠ {lang}: the review/fix pass introduced a language regression "
              f"({final_wrong_lang}) — reverting to the pre-review version.")
        output_text = pre_review_text
        lang_warnings.append(
            f"The AI review/fix pass introduced a language regression for '{lang}' and was "
            f"discarded; the pre-review version was kept instead."
        )
    if skip_review:
        lang_warnings.append("Semantic review was skipped for this large input; manually review business alignment before use.")
    lang_warnings = list(dict.fromkeys(lang_warnings))

    # Deterministic positioning guarantee (see note above commitment_chunk and
    # gender_neutral_chunk): this runs last, after every other step including the
    # review/fix pass, so both rules are guaranteed to be the literal first thing in
    # the document, verbatim, no matter what the model produced.
    guaranteed_header = []
    if commitment_chunk:
        guaranteed_header.append(commitment_chunk["content"].strip())
    if gender_neutral_chunk:
        guaranteed_header.append(gender_neutral_chunk["content"].strip())
    if guaranteed_header:
        output_text = "\n\n".join(guaranteed_header) + "\n\n" + output_text.lstrip()

    # GUARANTEE 6: a "Customise Prompt" instruction always renders as its own
    # clearly labeled, distinct section — unconditionally appended by code, never
    # left to the model's own compliance, and never conditioned on whether it
    # appears to have "already been covered" elsewhere in the document. That
    # unconditional rule is deliberate and non-negotiable: an earlier version of
    # this guarantee relied on a coverage-style check to decide whether the
    # customization needed to be added, and that exact shape of check is what let
    # a customization silently vanish from Kannada in an earlier round while
    # working fine for English and Hindi in the same request. Making this
    # unconditional is what actually closed that gap — a second, always-present
    # check can't have the same blind spot as the first one it's guarding against.
    #
    # What CAN change, safely, is how this guaranteed section is WORDED — nothing
    # about when or whether it renders. When the customization was woven into an
    # existing category (customization_category is set), the model was already
    # asked to apply its real effect within that section's own prose (see
    # synthesize_language_prompt's customization_block), so unconditionally
    # repeating the raw instruction again as a bare, unexplained second rule reads
    # like two competing instructions that happen to agree, rather than one
    # instruction with two touchpoints. Framing this section as an explicit
    # CONFIRMATION of what was already applied above makes that relationship
    # explicit instead of leaving the reader to infer it — but WITHOUT naming a
    # specific section title: the model's actual header wording (e.g. "Numbers,
    # Currency, and Prices") is free text the model chooses at synthesis time, and
    # customization_category is our own internal classification label (e.g.
    # "preserve_english"); the two are frequently different strings describing the
    # same thing, so pointing at "see {internal label}" made a claim about the
    # document's actual headings that this code has no way to guarantee is true.
    # Naming no section at all is the honest version of the same confirmation.
    # When nothing matched (customization_category is None), there is no "above"
    # to point to at all, so this section is the only place the rule exists and
    # must still read as a real, actionable instruction, not an empty confirmation
    # pointing at nothing.
    #
    # Appended last, after the review/fix pass, so nothing downstream can move,
    # reword, or drop it — same positioning principle as commitment_chunk and
    # gender_neutral_chunk above.
    if custom_instruction.strip():
        instruction_text = custom_instruction.strip()
        if customization_category:
            output_text = output_text.rstrip() + (
                "\n\n### Custom Instruction — Confirmation\n"
                f'You requested: "{instruction_text}" — this has been applied in the '
                f"appropriate section above."
            )
        else:
            output_text = output_text.rstrip() + (
                "\n\n### Custom Instruction\n"
                f'You requested: "{instruction_text}" — this did not match an existing rule '
                f"category, so it is included here as its own instruction:\n\n{instruction_text}"
            )

    return lang, output_text, lang_warnings


# A contrast conjunction after a clean-sounding opener ("looks good overall, but the
# numbers section is missing currency handling entirely") means a real caveat follows —
# the same class of problem _has_nearby_negation solves above for keyword triggers: a
# positive signal sitting next to a flip-word must not be read as positive. This list is
# deliberately small and literal (not "although", not "yet") to match exactly what was
# asked for, rather than guessing at a broader set.
_REVIEW_CONTRAST_CUES = ["but", "however", "though", "except", "aside from"]


def _review_found_issues(review_text: str) -> bool:
    """True if the review actually flagged something, false if it's a clean-bill-of-health
    response. The reviewer is instructed to say so in one line when nothing is wrong, so a
    short response containing one of these phrases (and no bullet-style findings) means
    skip the fix step — there's nothing to correct and re-running synthesis on a clean
    prompt only risks introducing a new problem where there wasn't one.

    A contrast conjunction overrides the clean-signal check entirely rather than being
    weighed against it: "looks good, but X" is exactly the shape a single real finding
    takes when the reviewer leads with a pleasantry, and no length or clean-phrase
    threshold should be able to out-vote that."""
    if not review_text or not review_text.strip():
        return False
    lowered = review_text.lower()
    if any(re.search(r'\b' + re.escape(cue) + r'\b', lowered) for cue in _REVIEW_CONTRAST_CUES):
        return True
    clean_signals = ["nothing wrong", "no issues", "no gaps", "looks good", "looks fine",
                      "no missing", "everything is covered", "no problems found", "no concerns"]
    is_short = len(review_text.strip()) < 200
    has_clean_signal = any(sig in lowered for sig in clean_signals)
    return not (is_short and has_clean_signal)


# ---------- CUSTOMISE PROMPT: per-generation instruction category matching ----------
# Lets someone add a one-off instruction the standard chunk library doesn't cover
# (e.g. "Price or currency should always be spoken in English throughout, in every
# language") without it either silently duplicating an existing rule category, or
# fighting it as a separate, potentially conflicting section.

# Maps a TAG_TRIGGERS domain to the chunk category that owns it, for reinforcing an
# existing category rather than creating a redundant new section for it.
_CUSTOMIZATION_TAG_TO_CATEGORY = {
    "pincode": "pincode_phone", "phone_number": "pincode_phone",
    "currency": "currency", "gold_weight": "gold_weight",
    "dates": "dates", "time_pronunciation": "time_pronunciation",
    "branch_names": "branch_names", "identity_documents": "identity_documents",
    "lending_insurance_terms": "lending_insurance_terms",
}

# The specific domains the deterministic safety-critical guarantee (Guarantee 2 in
# _generate_one_language) force-enforces — this is the subset of the mapping above
# whose enforcement can conflict with a user customization, since Guarantee 2 assumes
# in-language numbers/currency/identifiers are always correct.
_CUSTOMIZATION_GUARANTEE_DOMAINS = {
    "pincode": TAG_TRIGGERS["pincode"],
    "phone": TAG_TRIGGERS["phone_number"],
    "currency": TAG_TRIGGERS["currency"],
    "gold": TAG_TRIGGERS["gold_weight"],
}

_ENGLISH_PRESERVATION_CUES = [
    r"\bin english\b", r"\bstay(?:s|ing)? in english\b", r"\bkeep\b.{0,25}\bin english\b",
    r"\bnever translate\b", r"\balways\b.{0,25}\bin english\b", r"\bremains? in english\b",
    r"\bspoken\b.{0,15}\bin english\b", r"\bsay\b.{0,25}\bin english\b",
]

_HONORIFIC_CUES = ["honorific", " sir ", " madam ", "respectful address", "gender-neutral",
                   "gender neutral", "pronoun", "address the customer", "gender agnostic",
                   "gender-agnostic"]
_TONE_CUES = ["tone", "colloquial", "register", "casual", "formal style", "pacing"]

# A per-request "Customise Prompt" text box must never be able to override WHICH
# language is fundamentally being spoken — that's exactly what the always-on
# language_commitment guarantee exists to make un-overridable except by the
# business's own actual logic. A real production customization ("Speak only in
# Hindi, maintaining Hindi grammar and sentence structure") reached this far and got
# misclassified as an unrelated category ('preserve_english', because a companion
# clause about English happened to be in the same free-text box) instead of being
# rejected outright. This is checked FIRST, before any category classification
# runs, and independently of whichever language is currently being generated — a
# customization that names a spoken language and asserts it as the one to use is
# rejected for every requested language, not just the one it happens to name.
#
# This delegates to _unit_imposes_language_switch — the exact same detector used to
# strip the main synthesized body and to filter extraction-sourced custom notes
# (see _strip_language_switching_instructions and
# _filter_custom_notes_for_language_imposition above) — rather than its own separate
# pattern, so all three sources are provably governed by one mechanism, not three
# patches that happen to agree with each other.
def _customization_overrides_language_identity(instruction: str) -> bool:
    """True if the customization's own text amounts to asserting which spoken
    language the agent uses at all (e.g. "speak only in Hindi", "switch to English",
    "use Kannada throughout", or a speak_hi/speak_en-style field reference) — as
    opposed to a rule about how to say something WITHIN whichever language is
    already being generated. This must be rejected outright rather than classified
    into any category, matched or not."""
    return _unit_imposes_language_switch(instruction)


def _classify_customization(instruction: str) -> tuple:
    """Best-effort match of a free-text custom instruction to an existing chunk
    category, so it reinforces that category for this generation instead of creating a
    separate, potentially conflicting section. Falls back to no match (caller should
    let it stand on its own as a new section) when nothing fits.

    Returns (category_or_None, guarantee_domain_or_None). guarantee_domain is one of
    the deterministic safety-critical domains (pincode/phone/currency/gold) this
    instruction affects, if any — separate from category because a customization can
    be classified as 'preserve_english' (the category label) while still needing to
    stand down the 'currency' safety guarantee specifically (see
    _CUSTOMIZATION_GUARANTEE_DOMAINS' usage in _generate_one_language)."""
    if not instruction or not instruction.strip():
        return None, None
    text = f" {instruction.lower()} "

    guarantee_domain = None
    for domain, keywords in _CUSTOMIZATION_GUARANTEE_DOMAINS.items():
        if any(_keyword_present(text, kw) for kw in keywords):
            guarantee_domain = domain
            break

    if any(re.search(p, text) for p in _ENGLISH_PRESERVATION_CUES):
        return "preserve_english", guarantee_domain

    for tag, category in _CUSTOMIZATION_TAG_TO_CATEGORY.items():
        if any(_keyword_present(text, kw) for kw in TAG_TRIGGERS[tag]):
            return category, guarantee_domain

    if any(cue in text for cue in _HONORIFIC_CUES):
        return "honorifics", guarantee_domain
    if any(cue in text for cue in _TONE_CUES):
        return "colloquial_speech", guarantee_domain

    return None, guarantee_domain


def generate_language_prompts_multi(clean_business_logic: str, languages: list, chunks_file="chunks.json",
                                     model: str = DEFAULT_MODEL, max_retries: int = 3,
                                     max_workers: int = 4, custom_notes_by_language: dict = None,
                                     custom_instruction: str = "", agent_gender: str = "unspecified") -> dict:
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

    custom_instruction: optional free-text "Customise Prompt" instruction for this one generation
    (e.g. "Price or currency should always be spoken in English"), applied identically to every
    requested language via its own dedicated parameter into _generate_one_language — never merged
    into the custom_notes channel (an earlier design did this, and relying on synthesis fidelity
    plus a fuzzy per-unit coverage-ratio check meant for aggregate business notes, not a single
    UI instruction, made its presence unreliable: it appeared in 2 of 3 languages in the same real
    request and silently no-op'd for the third). It is now ALWAYS rendered as its own distinct,
    clearly labeled section, appended deterministically by code — see Guarantee 6 in
    _generate_one_language — never merged into or absorbed by an existing category's section, and
    never dependent on the model's own compliance for whether it appears at all. It is still
    matched against the existing chunk categories (see _classify_customization) so its ACTUAL
    EFFECT can additionally inform the relevant section's own prose where useful, but that's a
    bonus, not what guarantees its presence. Never persisted to chunks.json. A customization that
    amounts to overriding which language is spoken at all (e.g. "speak only in Hindi") is rejected
    outright rather than classified — see _customization_overrides_language_identity.

    agent_gender: optional centrally-determined agent gender for this business — "male", "female",
    or "unspecified" (see detect_agent_gender). Threaded identically into every requested language's
    generation so gender is decided ONCE, not independently reinterpreted per language.

    Returns {"prompts": {lang: text}, "warnings": {lang: [violation strings]}}.
    """
    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    custom_notes_by_language = dict(custom_notes_by_language or {})
    prompts = {}
    warnings = {}

    custom_instruction = (custom_instruction or "").strip()
    matched_category = None
    customization_guarantee_domain = None
    customization_rejected_warning = None

    if custom_instruction and _customization_overrides_language_identity(custom_instruction):
        # A per-request text box must never be able to touch WHICH language is spoken —
        # that's exactly what the always-on language_commitment guarantee exists to make
        # un-overridable except by the business's own actual logic. Reject the
        # customization outright, for every requested language, rather than attempting
        # to classify or apply any part of it.
        customization_rejected_warning = (
            f"Custom instruction ignored: \"{custom_instruction}\" attempts to override which "
            f"language is spoken. Customizations cannot override the language-commitment "
            f"guarantee — that is controlled only by the business's own logic and the language "
            f"you requested."
        )
        print(f"  ⚠ Rejected custom instruction — it attempts to override the spoken language "
              f"itself, which is not supported: {custom_instruction!r}")
        custom_instruction = ""
    elif custom_instruction:
        matched_category, customization_guarantee_domain = _classify_customization(custom_instruction)

    agent_gender_directive = _build_agent_gender_directive(agent_gender)

    effective_max_retries, skip_review = _adaptive_limits(len(clean_business_logic), max_retries)
    if skip_review:
        print(f"  ⚠ Large input ({len(clean_business_logic)} chars) — reducing retries to "
              f"{effective_max_retries} and skipping the review pass to stay within the time budget.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one_language, clean_business_logic, lang, all_chunks, model, effective_max_retries,
                             custom_notes_by_language.get(lang, ""), skip_review, customization_guarantee_domain,
                             custom_instruction, matched_category, agent_gender_directive): lang
            for lang in languages
        }
        for future in as_completed(futures):
            lang, output_text, lang_warnings = future.result()
            prompts[lang] = output_text
            if customization_rejected_warning:
                lang_warnings = [customization_rejected_warning] + lang_warnings
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
    "bengali": "BN", "bangla": "BN", "bn": "BN",
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


LANGUAGE_SCOPE_SYSTEM_PROMPT = """You are given a business's raw system prompt for a voice AI agent. Determine whether the business explicitly restricts which spoken languages the agent is allowed to use — for example "converse only in Hindi or English, other languages are not supported", "never speak any other language apart from Hindi, English, and Telugu", or "Kannada is the primary language, switch to English if asked".

If the business states an explicit, closed list of supported languages, output ONLY that list of language names, comma-separated, nothing else (e.g. "Hindi, English").

If the business does not state any such restriction anywhere (no mention of which languages it does or doesn't support), output exactly: NONE

Do not guess or infer a restriction from indirect signals. Only report a restriction that is explicitly and directly stated in the text."""


def detect_declared_language_scope(business_logic: str, model: str = DEFAULT_MODEL) -> list:
    """Reads what languages a business ACTUALLY authorizes, directly from its own
    prompt, rather than discovering the mismatch only after a document for an
    unauthorized language has already been generated and someone happens to notice.
    This is the fix for a recurring, distinct failure pattern seen across several
    businesses in this pipeline's own history — a business explicitly says something
    like "converse only in Hindi or English, other languages are not supported," and a
    request for a different language still produces a fully-formed document for it,
    because nothing ever checked the request against what the business itself
    declared. Returns a list of internal language codes if the business states an
    explicit restriction, or an empty list if it doesn't (in which case no scope
    warning should be shown — an unrestricted business supporting any requested
    language is the normal, expected case)."""
    model = _resolve_model(model)
    if not business_logic.strip():
        return []
    full_prompt = f"{LANGUAGE_SCOPE_SYSTEM_PROMPT}\n\n---BUSINESS PROMPT---\n{business_logic[:6000]}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=200)
    )
    result = (response.text or "").strip()
    if not result or result.upper() == "NONE":
        return []
    declared = []
    for name in result.split(","):
        code = LANGUAGE_NAME_MAP.get(name.strip().lower())
        if code and code not in declared:
            declared.append(code)
    return declared


# ---------- STAGE: Agent Gender — determined once, applied identically everywhere ----------

AGENT_GENDER_SYSTEM_PROMPT = """You are given a business's raw system prompt (or its already-cleaned business logic) for a voice AI agent. Determine whether the business explicitly states the AGENT's own gender anywhere — not the customer's gender, the agent's own.

Look for a direct statement of the agent's own gender: "You are female", "The agent is male", "refer to yourself in feminine verb forms", "Sona, a female representative", or an unambiguous first-person gendered framing tied to the agent's own persona.

A name alone (e.g. "Vinayak", "Sona", "Ravi", "Priya") is NOT a gender statement, even if the name is conventionally associated with one gender — many names are ambiguous, used across genders, or simply not a reliable signal, and this system must never guess or infer a gender from a name alone.

If the business explicitly states the agent is male, output exactly: MALE
If the business explicitly states the agent is female, output exactly: FEMALE
If the business does not explicitly state the agent's own gender anywhere, output exactly: UNSPECIFIED

Do not guess or infer from indirect signals like a name, a persona description with no gender word, or a tone description. Only report MALE or FEMALE if a real, direct, unambiguous statement of the agent's own gender exists in the text; otherwise report UNSPECIFIED."""


def detect_agent_gender(business_logic: str, model: str = DEFAULT_MODEL) -> str:
    """Determines the agent's gender ONCE, directly from the business's own prompt,
    the same way declared language scope is determined once rather than left to
    each language's own synthesis call to decide independently. This is the fix for
    a real production case: a business named its agent "Vinayak" but never stated a
    gender anywhere, and the English document asserted male, the Hindi document
    asserted female, and only the Kannada document correctly deferred — three
    languages in the same request disagreeing (two of them asserting OPPOSITE
    genders for the same persona) because each one's synthesis call independently
    invented or defaulted a gender, with no shared source of truth to agree on.

    Returns "male", "female", or "unspecified" — the single determination every
    requested language's generation must then use identically (see
    _build_agent_gender_directive)."""
    model = _resolve_model(model)
    if not business_logic.strip():
        return "unspecified"
    full_prompt = f"{AGENT_GENDER_SYSTEM_PROMPT}\n\n---BUSINESS PROMPT---\n{business_logic[:6000]}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=_build_config(model, max_output_tokens=20)
    )
    result = (response.text or "").strip().upper()
    if result == "MALE":
        return "male"
    if result == "FEMALE":
        return "female"
    return "unspecified"


def _build_agent_gender_directive(agent_gender: str) -> str:
    """Builds the single, fixed, language-agnostic directive injected identically
    into every requested language's synthesis call — the actual fix for gender
    being decided independently per language. The per-language agent_gender CHUNK
    in chunks.json still supplies the language-specific GRAMMATICAL MECHANICS
    (which verb endings, which pronoun, for each case); this directive supplies the
    single centrally-determined FACT of which case applies, so every language
    resolves that same chunk's conditional guidance the same way instead of each
    one guessing on its own."""
    agent_gender = (agent_gender or "unspecified").lower()
    if agent_gender == "male":
        return (
            "AGENT GENDER — determined once for this business from its own source prompt, "
            "identical for every language, do not reinterpret or override: the agent is MALE. "
            "Apply this consistently using this language's own male self-reference forms per "
            "the agent-gender guidance below."
        )
    if agent_gender == "female":
        return (
            "AGENT GENDER — determined once for this business from its own source prompt, "
            "identical for every language, do not reinterpret or override: the agent is FEMALE. "
            "Apply this consistently using this language's own female self-reference forms per "
            "the agent-gender guidance below."
        )
    return (
        "AGENT GENDER — determined once for this business from its own source prompt, identical "
        "for every language, do not reinterpret or override: this business's own prompt does NOT "
        "state the agent's gender. Do not assume, guess, or assert a gender for the agent — "
        "including never inferring one from the agent's name alone. Use neutral, deployment-"
        "configured, or gender-non-specific self-reference forms consistently, per the "
        "agent-gender guidance below."
    )


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
   - Specific number, date, or quantity CORRECTIONS tied to this business's own offerings, where the point is preventing a specific mispronunciation or mix-up (e.g. "Purnima Batch is thirteen days, never say twenty-three" — the value itself is almost incidental, the actual rule is a correction against a specific likely error)
   - Business-specific example sentences showing exact phrasing for THIS business's own numbers, prices, or terms, used to illustrate a pronunciation pattern
   - Any other explicit "say X, never Y" correction that is tied to this business's own facts, names, or terminology, not a generic language style preference

3. EXCLUDE PURE BUSINESS REFERENCE DATA, even when it contains numbers, dates, or lists — this is the single most common mistake to avoid. The test: does this sentence tell the agent HOW to correctly pronounce or phrase something (a genuine language rule, category 2 above), or does it tell the agent WHAT a business fact or value simply IS (pure information the agent needs to know, with no pronunciation angle at all)? Only the first kind belongs here. Concretely EXCLUDE, and never extract:
   - Rate cards, pricing tiers, fee percentages, or eligibility criteria (e.g. "loan rates start from 11.5 percent", "eligibility goes up to 80 percent of car value") — these are business facts the agent needs to know, not corrections about how to say a number.
   - Bonus, discount, or incentive tables tied to value ranges (e.g. an exchange-bonus table listing amounts for different price brackets) — same reasoning, this is what to say, not how to say it.
   - Lists of cities, branches, locations, or areas the business operates in — these are business scope facts, not language rules, even if there's a pronunciation instruction elsewhere that references "these city names."
   - Policy facts, timelines, counts, or operational figures (e.g. "cars are inspected at 15 refurbishment centers", "5 day return policy", "loan approval takes 24 to 48 hours") — pure business information.
   These all belong exclusively in business logic, which the agent already has access to separately — duplicating them here means the same reference data gets copied into every single language's document identically, which is pure bloat with zero language-specific value added. If a sentence is just informing the agent of a fact or figure, with no genuine correction or pronunciation guidance attached, leave it out even if it involves numbers.

EXTRACT THE PHRASING RULE, DISCARD THE FACT IT'S ATTACHED TO — a single sentence often mixes a real communication instruction with a specific business value, and these need to be pulled apart rather than kept or dropped as one unit. If a sentence tells the agent HOW to present a whole CATEGORY of business information (regardless of what the specific number happens to be), extract only that general presentation rule, generalized and with the specific figures stripped out, and discard the actual value entirely as excluded business data per rule 3 above.
   - Example: "For all pricing-related responses, just say the price in lakhs without the unit (for example, five lakh fifteen thousand)" — KEEP the rule "speak prices in lakhs, without a currency unit word" as a language rule; the "five lakh fifteen thousand" example is fine to keep too since it's illustrating a pattern, not stating this business's actual price.
   - Example: "Loan rates start from 11.5 percent" — there is no separable presentation rule here, it's just the value itself, so exclude the whole sentence per rule 3.
   - Example: "Exchange bonus is always quoted as a round number in thousands, never with paise or decimal precision" — KEEP this as a presentation rule (how to phrase any bonus figure), even though the specific bonus amounts in an accompanying table must still be excluded per rule 3.
   The output rule of thumb: if you can state a version of the sentence that would still make sense for a DIFFERENT number than the one given, that's a phrasing rule, keep it generalized. If the sentence only makes sense because of the specific value it names, that value is the business fact, exclude it.

COMPRESS MECHANICALLY-DERIVABLE LISTS — this matters and is easy to get wrong: a source sometimes writes out an exhaustive list that is really just ONE generic pattern repeated many times (e.g. a table pronouncing every year from 2015 to 2026 individually, when every entry follows the exact same "two thousand + [unit word]" pattern with no exceptions). That is NOT a business-specific fact, it's a generic language rule that happens to be formatted as a long list — state the pattern once with 1-2 examples, and drop the rest of the mechanically-identical entries. Do NOT apply this compression to anything genuinely irregular, business-specific, or containing even one exception (e.g. a numbering pattern that breaks for one specific value, a business's own specific correction like "never say twenty-three") — those must stay complete and verbatim. When in doubt about whether a list is a repeated pattern or contains real exceptions, keep it in full; only compress when you are certain every entry is a mechanical restatement of the same rule.

For all languages, compress generic number/currency lookup lists into their reading patterns and meaningful exceptions with at most one source example. Preserve business-specific corrections and identifier reading rules.

For all languages, compress generic date/time guidance into a short pattern with meaningful exceptions and at most one source example. Do not copy minute-by-minute or year-by-year example lists. Preserve explicit business-specific corrections and exact values.

Output ONLY the extracted custom language/speech rules (item 2), removing all flow-control content (item 1) entirely. Do not summarize, paraphrase, shorten, or reword the custom rules — copy them close to verbatim so no specific figure or spelling is lost, EXCEPT for mechanically-derivable lists per the compression rule above. If there is genuinely nothing that qualifies as a custom language rule in this text, output nothing at all (an empty response is correct and expected in that case, do not invent content to fill space).

NEVER PREFACE THE OUTPUT WITH ANY NARRATING OR CONVERSATIONAL FRAMING — no "Here are the custom language and speech rules extracted from the document:", no "Sure, here is...", no "The following rules were found:", no summary sentence of what you did before or after the rules, and no closing remark. This output is fed directly into a customer-facing voice AI document as-is, verbatim — any such framing sentence would ship into that document exactly as written, not as a description of your work. Output rule content only, starting on the very first line, or nothing at all."""


# Matches a leading conversational preamble line/sentence (e.g. "Here are the
# custom language and speech rules extracted from the document:", "Sure, here is
# the extracted content:", "The following rules were found:") that the extraction
# model sometimes prepends before the actual content despite
# CUSTOM_LANGUAGE_NOTES_SYSTEM_PROMPT explicitly forbidding it. Guarantee 1
# downstream (_generate_one_language) has no way to distinguish this from genuine
# custom-notes content — it faithfully preserves whatever custom_notes contains —
# so it must be caught here, at the source, before it's ever treated as real
# content.
#
# A preamble is reliably a SHORT leading line ending in a colon (anchored to the
# very start of the string, never mid-document) that narrates what the model is
# about to do rather than stating a rule — gated on a cue-phrase list rather than
# one fixed sentence shape, since "here are the rules:", "the following were
# found:", and "I've extracted:" all narrate the same way in different words. A
# genuine custom rule's own leading colon-terminated clause (e.g. "Sacred names
# policy: always write in Devanagari") never matches any cue below, so this can't
# mistake real content for a preamble.
_EXTRACTION_PREAMBLE_LEADING_LINE = re.compile(r'^\s*([^\n]{0,160}?):\s*\n?')
_EXTRACTION_PREAMBLE_CUES = [
    "here are", "here is", "here's", "the following", "below are", "below is",
    "i've extracted", "i have extracted", "were found", "was found", "found the following",
    "extracted from the document", "extracted content", "extracted rules", "extracted custom",
]


def _strip_extraction_preamble(text: str) -> tuple:
    """Deterministic defense-in-depth for the extraction-preamble leak — the same
    'don't just ask nicely, strip it deterministically' pattern already used
    elsewhere in this file (see _strip_customization_meta_leak,
    _strip_language_switching_instructions). Returns (cleaned_text, count_removed —
    0 or 1, since a preamble can only ever be the leading line)."""
    if not text:
        return text, 0
    match = _EXTRACTION_PREAMBLE_LEADING_LINE.match(text)
    if not match:
        return text, 0
    leading_line = match.group(1).lower()
    if not any(cue in leading_line for cue in _EXTRACTION_PREAMBLE_CUES):
        return text, 0
    return text[match.end():].lstrip(), 1


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
        result = (response.text or "").strip()
        if result:
            result, preamble_removed = _strip_extraction_preamble(result)
            if preamble_removed:
                print("  ⚠ Stripped a conversational preamble from extracted custom notes.")
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
