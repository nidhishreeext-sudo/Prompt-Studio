"""Output flow for flow-JSON inputs.

When the input is a guided-conversation-flow export, Prompt Studio returns the
same flow with its structure intact: in "extract" mode the global prompt and
each eligible node instruction are replaced by their distilled business logic,
and in every mode the generated language prompts are attached under
data.prompts. Nothing else in the flow changes, and the input is never mutated
(every change is made on a deep copy).

flow_check.py then proves the rewrite did not damage the flow.
"""

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import main
from main import LANGUAGE_NAME_MAP, _strip_extraction_preamble, _unescape_markdown, extract_text_from_flow_json

# Bounded so a large flow doesn't fire dozens of simultaneous Gemini calls
# (rate limits, and the gunicorn --timeout 600 budget the whole request shares).
EXTRACT_MAX_WORKERS = 5

# A distilled field shorter than this fraction of its original is treated as
# over-summarized (or blocked) and the original string is kept instead (D6).
MIN_DISTILLED_RATIO = 0.40

# Instructions that are placeholders, not content worth a Gemini call.
_TRIVIAL_INSTRUCTIONS = {"", "."}

# _generate_one_language returns this prefix instead of a usable prompt when
# every attempt came back in the wrong language. Such output is never attached.
GENERATION_FAILED_PREFIX = "⚠ GENERATION FAILED FOR"

# Internal language code -> BCP-47 tag used in the flow's per-language keys.
# hi-IN, te-IN, kn-IN, ta-IN and mr-IN appear in flow_guard's fixture; the
# others follow the same <iso639>-IN pattern (Odia's ISO 639-1 code is "or").
_LANG_CODE_TO_BCP47 = {
    "HI": "hi-IN", "EN": "en-IN", "KN": "kn-IN", "TA": "ta-IN", "ML": "ml-IN",
    "GU": "gu-IN", "MR": "mr-IN", "TE": "te-IN", "OD": "or-IN", "BN": "bn-IN",
}


def language_prompt_key(lang_code: str) -> str:
    """data.prompts key for a generated language prompt.

    Convention taken from flow_guard/tests/fixtures/synthetic_flow.json, where
    data.prompts holds "default" plus per-language prompts keyed "lang:te-IN"
    and "lang:hi-IN" (the same "lang:<tag>" form the node and edge overlays use).
    """
    return f"lang:{_LANG_CODE_TO_BCP47.get(lang_code, lang_code.lower())}"


@dataclass
class FlowField:
    """One text field of the flow that the distill step may rewrite."""
    path: str              # flow_guard's path: "data.prompt" or "nodes[<id>].instruction"
    label: str             # readable location for field notes
    text: str              # the original string, byte-for-byte
    node_key: object = None  # list index (or dict key) of the node; None for data.prompt
    language_node: bool = False

    def write(self, flow: dict, value: str) -> None:
        data = flow["data"]
        if self.node_key is None:
            data["prompt"] = value
        else:
            data["conversationNodes"]["nodes"][self.node_key]["instruction"] = value


@dataclass
class FlowDistillation:
    business_logic: str
    rewritten: dict = field(default_factory=dict)   # {FlowField.path: (FlowField, distilled text)}
    notes: list = field(default_factory=list)
    extract_calls: int = 0


@dataclass
class OutputFlow:
    flow: dict
    rewritten_paths: list      # every path whose text Prompt Studio replaced
    attached_keys: list        # data.prompts keys Prompt Studio added or replaced
    created_prompts: bool      # True when the input had no data.prompts at all
    notes: list


def _note(path: str, label: str, status: str, detail: str) -> dict:
    return {"field": path, "label": label, "status": status, "detail": detail}


def _node_items(flow: dict) -> list:
    nodes = ((flow.get("data") or {}).get("conversationNodes") or {}).get("nodes") or []
    if isinstance(nodes, dict):
        return list(nodes.items())
    return [(i, n) for i, n in enumerate(nodes)]


def is_language_node(node: dict) -> bool:
    """Nodes named after a language (Hindi, Kannada, ...) are already language layers."""
    return (node.get("name") or "").strip().lower() in LANGUAGE_NAME_MAP


def _node_fields(flow: dict, language_nodes: bool) -> list:
    fields = []
    for key, node in _node_items(flow):
        if not isinstance(node, dict) or is_language_node(node) != language_nodes:
            continue
        instruction = node.get("instruction")
        if not isinstance(instruction, str) or instruction.strip() in _TRIVIAL_INSTRUCTIONS:
            continue
        name = node.get("name") or node.get("id")
        fields.append(FlowField(path=f"nodes[{node.get('id')}].instruction",
                                label=f"Node '{name}' instruction", text=instruction,
                                node_key=key, language_node=language_nodes))
    return fields


def list_rewritable_fields(flow: dict) -> list:
    """data.prompt plus every non-empty, non-trivial instruction on a node that
    is not language-named. These are the only fields the output flow may change."""
    fields = []
    prompt = (flow.get("data") or {}).get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        fields.append(FlowField(path="data.prompt", label="Global prompt (data.prompt)", text=prompt))
    return fields + _node_fields(flow, language_nodes=False)


def list_language_node_fields(flow: dict) -> list:
    return _node_fields(flow, language_nodes=True)


def _rejection_reason(original: str, distilled, error) -> str | None:
    if error is not None:
        return f"the extraction call failed ({type(error).__name__}: {str(error)[:120]})"
    if not distilled or not distilled.strip():
        return "extraction returned nothing (for example a safety-blocked response)"
    base = len(_unescape_markdown(original).strip())
    if base and len(distilled.strip()) < MIN_DISTILLED_RATIO * base:
        return (f"extraction returned only {len(distilled.strip())} of {base} characters "
                f"(below the {int(MIN_DISTILLED_RATIO * 100)}% minimum), so it was likely over-summarized")
    return None


def distill_flow(flow_json: dict, model: str, extract_fn=None) -> FlowDistillation:
    """Mode "extract" for a flow: distill each field separately.

    Returns the flattened business_logic for the downstream pipeline (same
    layout as extract_text_from_flow_json) plus the accepted rewrites.
    Language-named nodes are distilled too, but only for business_logic: their
    instruction is never written back into the output flow.
    """
    extract = extract_fn or main.extract_business_logic
    rewritable = list_rewritable_fields(flow_json)
    jobs = rewritable + list_language_node_fields(flow_json)

    results = []
    if jobs:
        with ThreadPoolExecutor(max_workers=min(EXTRACT_MAX_WORKERS, len(jobs))) as executor:
            futures = [executor.submit(extract, _unescape_markdown(f.text), model=model) for f in jobs]
            for f, future in zip(jobs, futures):
                try:
                    # A leading "Here is the cleaned prompt:" must never land in a
                    # deployable instruction. Stripped before the length checks, so
                    # a preamble-only response counts as empty.
                    distilled, _ = _strip_extraction_preamble(future.result())
                    results.append((f, distilled, None))
                except Exception as exc:  # one failed field must not sink the rest
                    results.append((f, None, exc))
        if all(err is not None for _, _, err in results):
            raise results[0][2]  # nothing was distilled: fail like the single-call path did

    shaped = copy.deepcopy(flow_json)  # the flow as the downstream pipeline should see it
    distillation = FlowDistillation(business_logic="", extract_calls=len(jobs))
    for f, distilled, err in results:
        if f.language_node:
            text = (distilled or "").strip()
            f.write(shaped, text)
            detail = "Language-named node, left untouched in the output flow."
            if err is not None:
                detail += f" Its extraction failed ({type(err).__name__}), so it was left out of the business logic."
            else:
                detail += f" Its distilled text ({len(text)} chars) was used only for the business logic."
            distillation.notes.append(_note(f.path, f.label, "language_node_untouched", detail))
            continue
        reason = _rejection_reason(f.text, distilled, err)
        if reason:
            distillation.notes.append(_note(f.path, f.label, "kept_original",
                                            f"Original kept byte-for-byte: {reason}."))
            continue
        text = distilled.strip()
        if text == _unescape_markdown(f.text).strip():
            distillation.notes.append(_note(f.path, f.label, "unchanged",
                                            "No language rules found; original kept byte-for-byte."))
            continue
        f.write(shaped, text)
        distillation.rewritten[f.path] = (f, text)
        distillation.notes.append(_note(f.path, f.label, "rewritten",
                                        f"Distilled to business logic ({len(f.text)} -> {len(text)} chars)."))

    distillation.business_logic = extract_text_from_flow_json(shaped)
    return distillation


def build_output_flow(flow_json: dict, distillation: FlowDistillation | None, language_prompts: dict) -> OutputFlow:
    """Deep copy of the input with the accepted rewrites applied and each
    successfully generated language prompt attached under data.prompts."""
    out = copy.deepcopy(flow_json)
    notes = []
    rewritten_paths = []

    if distillation is None:
        # Scoped mode: no rewriting. Language nodes still deserve a note so the
        # user can see they were recognized and deliberately left alone.
        for f in list_language_node_fields(flow_json):
            notes.append(_note(f.path, f.label, "language_node_untouched",
                               "Language-named node, left untouched in the output flow."))
    else:
        for path, (f, text) in distillation.rewritten.items():
            f.write(out, text)
            rewritten_paths.append(path)

    data = out["data"]
    created_prompts = data.get("prompts") is None
    attached_keys = []
    usable = {lang: text for lang, text in (language_prompts or {}).items()
              if isinstance(text, str) and text.strip() and not text.startswith(GENERATION_FAILED_PREFIX)}
    for lang in language_prompts or {}:
        if lang not in usable:
            notes.append(_note(f"data.prompts.{language_prompt_key(lang)}", f"{lang} language prompt", "not_attached",
                               "Generation ended in an error state, so nothing was attached for this language."))

    if usable and not created_prompts and not isinstance(data.get("prompts"), dict):
        notes.append(_note("data.prompts", "Language prompts", "not_attached",
                           "data.prompts in the input is not an object, so no language prompt was attached."))
        usable = {}
    if usable and created_prompts:
        data["prompts"] = {}

    for lang, text in usable.items():
        key = language_prompt_key(lang)
        path = f"data.prompts.{key}"
        if key in data["prompts"]:
            if data["prompts"][key] != text:
                rewritten_paths.append(path)
            notes.append(_note(path, f"{lang} language prompt", "replaced",
                               f"Replaced the existing '{key}' prompt with the generated {lang} prompt."))
        else:
            notes.append(_note(path, f"{lang} language prompt", "attached",
                               f"Generated {lang} prompt attached as data.prompts['{key}']."))
        data["prompts"][key] = text
        attached_keys.append(key)

    return OutputFlow(flow=out, rewritten_paths=rewritten_paths, attached_keys=attached_keys,
                      created_prompts=created_prompts and bool(attached_keys), notes=notes)
