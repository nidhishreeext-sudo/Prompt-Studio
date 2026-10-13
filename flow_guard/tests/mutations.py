"""Helpers that corrupt a flow the way a bad import/export tool would.

Every helper takes a flow document (full export envelope), returns a mutated
deep copy, and never touches its argument. Targets are located generically
(first node with two edges, the switch node, a Telugu string...) so the same
mutations run against the synthetic fixture and the real sample. A helper
raises `NoTarget` when a flow has nothing to mutate; tests turn that into a skip.
"""

from __future__ import annotations

import copy
import re


class NoTarget(LookupError):
    pass


def _copy(doc: dict) -> dict:
    return copy.deepcopy(doc)


def conv(doc: dict) -> dict:
    return doc["data"]["conversationNodes"]


def nodes(doc: dict) -> list[dict]:
    return conv(doc)["nodes"]


def _edge_groups(node: dict):
    for group in ("edges", "goBackEdges", "jumpToEdges"):
        for edge in node.get(group) or []:
            yield group, edge


def _first(iterable, what: str):
    for item in iterable:
        return item
    raise NoTarget(what)


def node_with_distinct_edges(doc: dict, node_type: str = "conversation") -> dict:
    """First node of a type with >= 2 edges leading to different nodes."""
    return _first((n for n in nodes(doc) if n.get("type") == node_type
                   and len({e.get("destinationNodeId") for e in n.get("edges") or []}) >= 2),
                  f"{node_type} node with two edges to different nodes")


def edge_with_condition(doc: dict) -> tuple[dict, dict]:
    return _first(((n, e) for n in nodes(doc) for e in n.get("edges") or []
                   if ((e.get("transitionCondition") or {}).get("prompt") or "").strip()),
                  "edge with a non-empty condition")


# --- mutations ---------------------------------------------------------------

def identity(doc: dict) -> dict:
    return _copy(doc)


def reverse_nodes(doc: dict) -> dict:
    out = _copy(doc)
    nodes(out).reverse()
    return out


def _rename(doc: dict, mapping: dict[str, str], skip_edge: str | None = None) -> dict:
    out = _copy(doc)
    c = conv(out)
    if c.get("startNodeId") in mapping:
        c["startNodeId"] = mapping[c["startNodeId"]]
    for node in nodes(out):
        node["id"] = mapping.get(node["id"], node["id"])
        for _group, edge in _edge_groups(node):
            if edge.get("id") == skip_edge:
                continue
            if edge.get("destinationNodeId") in mapping:
                edge["destinationNodeId"] = mapping[edge["destinationNodeId"]]
    return out


def rename_ids(doc: dict) -> dict:
    return _rename(doc, {n["id"]: f"renamed-{i}-{n['id'][::-1]}" for i, n in enumerate(nodes(doc))})


def rename_ids_leave_dangling(doc: dict) -> dict:
    _node, edge = _first(((n, e) for n in nodes(doc) for e in n.get("edges") or [] if e.get("destinationNodeId")),
                         "any edge")
    return _rename(doc, {n["id"]: f"renamed-{i}-{n['id'][::-1]}" for i, n in enumerate(nodes(doc))},
                   skip_edge=edge["id"])


def shift_ids(doc: dict) -> dict:
    """Each node's content now sits under the next node's id; references unchanged."""
    out = _copy(doc)
    ids = [n["id"] for n in nodes(out)]
    for i, node in enumerate(nodes(out)):
        node["id"] = ids[(i + 1) % len(ids)]
    return out


def swap_edge_destinations(doc: dict) -> dict:
    out = _copy(doc)
    node = node_with_distinct_edges(out)
    edges = node["edges"]
    first = edges[0]
    second = _first((e for e in edges[1:] if e["destinationNodeId"] != first["destinationNodeId"]), "second edge")
    first["destinationNodeId"], second["destinationNodeId"] = second["destinationNodeId"], first["destinationNodeId"]
    return out


def delete_edge(doc: dict) -> dict:
    out = _copy(doc)
    node_with_distinct_edges(out)["edges"].pop()
    return out


def add_edge(doc: dict) -> dict:
    out = _copy(doc)
    node = node_with_distinct_edges(out)
    target = _first((n["id"] for n in nodes(out) if n.get("type") == "end"), "end node")
    node["edges"].append({
        "id": "edge-injected-by-tool",
        "destinationNodeId": target,
        "transitionCondition": {"type": "prompt", "prompt": "The caller asks to speak to a manager."},
        "transitionMessage": "",
        "overlays": {},
        "translationReviews": {},
        "goBackToPrevious": False,
    })
    return out


def change_condition_word(doc: dict) -> dict:
    out = _copy(doc)
    _node, edge = edge_with_condition(out)
    cond = edge["transitionCondition"]
    word = _first((m for m in re.finditer(r"[A-Za-z]{4,}", cond["prompt"])), "a word in the condition")
    replacement = "never" if word.group(0).lower() != "never" else "always"
    cond["prompt"] = cond["prompt"][: word.start()] + replacement + cond["prompt"][word.end():]
    return out


def whitespace_condition(doc: dict) -> dict:
    out = _copy(doc)
    _node, edge = edge_with_condition(out)
    cond = edge["transitionCondition"]
    cond["prompt"] = "  " + cond["prompt"].replace(" ", "   ", 1) + " \n"
    return out


def change_node_type(doc: dict) -> dict:
    out = _copy(doc)
    node = _first((n for n in nodes(out) if n.get("type") == "conversation"), "conversation node")
    node["type"] = "end"
    return out


def change_start(doc: dict) -> dict:
    out = _copy(doc)
    c = conv(out)
    c["startNodeId"] = _first((n["id"] for n in nodes(out) if n["id"] != c["startNodeId"] and n.get("type") != "end"),
                              "another node")
    return out


def swap_switch_cases(doc: dict) -> dict:
    out = _copy(doc)
    for node in nodes(out):
        if node.get("type") != "switch":
            continue
        dest = {e["id"]: e.get("destinationNodeId") for e in node.get("edges") or []}
        cases = node.get("switchCases") or {}
        keys = list(cases)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if dest.get(cases[a]) != dest.get(cases[b]):
                    cases[a], cases[b] = cases[b], cases[a]
                    return out
    raise NoTarget("switch node with two cases routing to different nodes")


def remove_overlay_language(doc: dict) -> dict:
    out = _copy(doc)
    node = _first((n for n in nodes(out) if any(k.startswith("lang:") for k in (n.get("overlays") or {}))),
                  "node with a language overlay")
    key = sorted(k for k in node["overlays"] if k.startswith("lang:"))[0]
    del node["overlays"][key]
    return out


def drop_placeholder(doc: dict, name: str = "devotee_name") -> dict:
    out = _copy(doc)
    token = "{{" + name + "}}"
    node = _first((n for n in nodes(out) if token in ((n.get("onEnterSay") or {}).get("message") or "")),
                  f"onEnterSay containing {token}")
    node["onEnterSay"]["message"] = node["onEnterSay"]["message"].replace(token, "").replace("  ", " ")
    return out


def change_time_in_prompt(doc: dict) -> dict:
    out = _copy(doc)
    prompt = out["data"].get("prompt") or ""
    m = _first(re.finditer(r"\b(\d{1,2})\s*(AM|PM|am|pm)\b", prompt), "a time in data.prompt")
    new_hour = str(int(m.group(1)) % 12 + 1)
    out["data"]["prompt"] = prompt[: m.start(1)] + new_hour + prompt[m.end(1):]
    return out


def change_amount_in_prompt(doc: dict) -> dict:
    out = _copy(doc)
    prompt = out["data"].get("prompt") or ""
    m = _first(re.finditer(r"(?:₹|Rs\.?)\s*(\d[\d,]*)", prompt), "an amount in data.prompt")
    out["data"]["prompt"] = prompt[: m.start(1)] + "999" + prompt[m.end(1):]
    return out


def _telugu_string_slot(doc: dict):
    """(container, key) of the first string holding Telugu text."""
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and re.search(r"[ఀ-౿]{3}", v):
                    yield obj, k
                else:
                    yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)
    return _first(walk(doc["data"]), "Telugu text")


def corrupt_telugu_fffd(doc: dict) -> dict:
    out = _copy(doc)
    obj, key = _telugu_string_slot(out)
    text = obj[key]
    i = re.search(r"[ఀ-౿]", text).start()
    obj[key] = text[:i] + "��" + text[i + 2:]
    return out


def corrupt_telugu_mojibake(doc: dict) -> dict:
    out = _copy(doc)
    obj, key = _telugu_string_slot(out)
    obj[key] = obj[key].encode("utf-8").decode("latin-1")
    return out


def _longest_instruction_node(doc: dict) -> dict:
    node = max(nodes(doc), key=lambda n: len(n.get("instruction") or ""))
    if len(node.get("instruction") or "") < 40:
        raise NoTarget("an instruction")
    return node


def truncate_instruction(doc: dict) -> dict:
    out = _copy(doc)
    node = _longest_instruction_node(out)
    node["instruction"] = node["instruction"][: len(node["instruction"]) // 2]
    return out


def reword_instruction(doc: dict) -> dict:
    """A rewording that keeps every token (numbers, placeholders...)."""
    out = _copy(doc)
    node = _longest_instruction_node(out)
    text = node["instruction"]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) >= 2:
        sentences[0], sentences[1] = sentences[1], sentences[0]
        node["instruction"] = " ".join(sentences)
    else:
        node["instruction"] = "Please note: " + text
    return out


def nudge_meaning_change(doc: dict) -> dict:
    out = _copy(doc)
    node = _first((n for n in nodes(out) if "exactly one nudge" in (n.get("instruction") or "")),
                  "instruction containing 'exactly one nudge'")
    node["instruction"] = node["instruction"].replace("exactly one nudge", "up to two nudges", 1)
    return out


def reset_positions(doc: dict) -> dict:
    out = _copy(doc)
    for node in nodes(out):
        if "position" in node:
            node["position"] = {"x": 0, "y": 0}
    if "startNodePosition" in conv(out):
        conv(out)["startNodePosition"] = {"x": 0, "y": 0}
    return out


def nodes_to_dict(doc: dict) -> dict:
    out = _copy(doc)
    conv(out)["nodes"] = {n["id"]: n for n in nodes(out)}
    return out


def reorder_edges(doc: dict) -> dict:
    out = _copy(doc)
    node_with_distinct_edges(out)["edges"].reverse()
    return out


def add_unknown_key(doc: dict) -> dict:
    out = _copy(doc)
    nodes(out)[0]["importedByToolX"] = True
    return out


def rename_global_variable(doc: dict) -> dict:
    out = _copy(doc)
    variables = conv(out).get("globalVariables") or []
    var = _first(iter(variables), "a global variable")
    var["name"] = var["name"] + "_renamed"
    return out


# --- token-multiset regressions ---------------------------------------------

_KB_HEADING = "## Seva Knowledge Base"
_KB_ROW_RE = re.compile(r"^\d+\. \*\*.*$", re.MULTILINE)


def _kb_section(prompt: str) -> tuple[int, int]:
    start = prompt.find(_KB_HEADING)
    if start < 0:
        raise NoTarget("a '## Seva Knowledge Base' section in data.prompt")
    nxt = prompt.find("\n## ", start + len(_KB_HEADING))
    return start, (nxt if nxt >= 0 else len(prompt))


def change_one_kb_amount(doc: dict, amount: str = "1,008", new: str = "1,108") -> dict:
    """Change ONE of several occurrences of an amount inside the Seva Knowledge
    Base (the third one, so neither the first nor the last)."""
    out = _copy(doc)
    prompt = out["data"]["prompt"]
    start, end = _kb_section(prompt)
    hits = [m for m in re.finditer(r"₹" + re.escape(amount) + r"(?![\d,])", prompt[start:end])]
    if len(hits) < 3:
        raise NoTarget(f"three occurrences of ₹{amount} in the Knowledge Base")
    m = hits[2]
    s, e = start + m.start(), start + m.end()
    out["data"]["prompt"] = prompt[:s] + "₹" + new + prompt[e:]
    return out


def shuffle_kb_rows(doc: dict) -> dict:
    """Reverse the order of the seva rows; every row (and amount) stays intact."""
    out = _copy(doc)
    prompt = out["data"]["prompt"]
    start, end = _kb_section(prompt)
    section = prompt[start:end]
    rows = list(_KB_ROW_RE.finditer(section))
    if len(rows) < 2:
        raise NoTarget("two seva rows in the Knowledge Base")
    block_start, block_end = rows[0].start(), rows[-1].end()
    block = section[block_start:block_end]
    lines = block.split("\n")
    row_lines = [ln for ln in lines if _KB_ROW_RE.match(ln)]
    if len(row_lines) != len(lines):
        raise NoTarget("contiguous seva rows")
    shuffled = "\n".join(reversed(row_lines))
    out["data"]["prompt"] = prompt[: start + block_start] + shuffled + prompt[start + block_end:]
    return out


_PLACEHOLDER_FORMS = {
    "curly": lambda name: r"\{\{\s*" + re.escape(name) + r"\s*\}\}",
    "square": lambda name: r"\\?\[\\?\[" + re.escape(name) + r"\]\]",
}


def _string_slots(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                yield obj, k
            else:
                yield from _string_slots(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                yield obj, i
            else:
                yield from _string_slots(v)


def delete_one_repeated_placeholder(doc: dict, name: str = "devotee_name") -> dict:
    """In a text field where the placeholder occurs several times, delete one
    occurrence (the last). Prefers {{name}}; falls back to [[name]] markers."""
    out = _copy(doc)
    for form in ("curly", "square"):
        pattern = re.compile(_PLACEHOLDER_FORMS[form](name))
        for container, key in _string_slots(out["data"]):
            hits = list(pattern.finditer(container[key]))
            if len(hits) >= 2:
                m = hits[-1]
                container[key] = container[key][: m.start()] + container[key][m.end():]
                return out
    raise NoTarget(f"a text field with {name} at least twice")


# --- list markers and time formats ------------------------------------------------

def kb_rows_to_bullets(doc: dict) -> dict:
    """Turn the numbered seva list ("1. **...") into bullets ("- **..."); text unchanged otherwise."""
    out = _copy(doc)
    prompt = out["data"]["prompt"]
    start, end = _kb_section(prompt)
    section, n = re.subn(r"(?m)^\d+\. ", "- ", prompt[start:end])
    if n < 2:
        raise NoTarget("a numbered list in the Knowledge Base")
    out["data"]["prompt"] = prompt[:start] + section + prompt[end:]
    return out


def rewrite_time(doc: dict, old: str = "10 AM", new: str = "10:00 AM") -> dict:
    """Replace the first occurrence of a time in data.prompt."""
    out = _copy(doc)
    prompt = out["data"].get("prompt") or ""
    m = re.search(r"(?<![\d:])" + re.escape(old) + r"\b", prompt)
    if not m:
        raise NoTarget(f"'{old}' in data.prompt")
    out["data"]["prompt"] = prompt[: m.start()] + new + prompt[m.end():]
    return out


def change_inline_number(doc: dict) -> dict:
    """Change a number inside a sentence (not a list marker, amount or time) in data.prompt."""
    from flow_guard.tokens import token_occurrences

    out = _copy(doc)
    raw = out["data"].get("prompt") or ""
    text, occ = token_occurrences(raw)
    covered = [(s, e) for kind, _t, s, e in occ if kind in ("currency", "time")]
    for kind, token, s, e in occ:
        if kind == "number" and not any(cs <= s < ce for cs, ce in covered):
            # Locate the same occurrence in the raw text by its neighbourhood.
            probe = text[max(0, s - 20): e + 20]
            i = raw.find(probe)
            if i < 0:
                continue
            j = i + (s - max(0, s - 20))
            out["data"]["prompt"] = raw[:j] + str(int(token.split(".")[0]) + 1) + raw[j + (e - s):]
            return out
    raise NoTarget("a number inside a sentence of data.prompt")
