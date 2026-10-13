"""B2-B12. Deterministic pair checks between an input flow and an output flow.

Every check works in "input id space": paths and node_id fields use the input
node ids, and output ids are translated through the node mapping first.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .loader import Flow, condition_text, edge_destination, edges_of
from .matcher import Mapping, edge_signature, signature_hash
from .model import Finding
from .policy import Policy
from .structure import reachable_from_start
from .tokens import collapse_ws, corruption_reasons, normalize_text, strip_tokens, token_delta

# Keys whose string values are human-language text (sent to text review).
TEXT_KEYS = {"instruction", "message", "description", "transitionMessage", "prompt",
             "initialMessage", "languagePrompt", "text", "content", "greeting"}

NODE_HANDLED = {"id", "name", "type", "globalNode", "overlays", "onEnterSay", "edges", "position",
                "instruction", "allowGoBack", "goBackEdges", "jumpToEdges",
                "switchExpression", "switchDefaultEdgeId", "switchCases"}
EDGE_HANDLED = {"id", "destinationNodeId", "transitionCondition", "transitionMessage", "overlays", "goBackToPrevious"}
CONV_HANDLED = {"startNodeId", "nodes", "startNodePosition", "globalVariables"}
DATA_HANDLED = {"conversationNodes", "prompt", "initialMessage", "prompts", "supportedModes"}
ENVELOPE_HANDLED = {"data", "exportedAt"}

_SENTENCE_RE = re.compile(r"(?<=[.!?।|])\s+")


@dataclass
class TextChange:
    """A text field whose change rules cannot classify; the judge decides."""
    group: str              # batching key: input node id, or "__global__"
    node_id: str | None
    path: str
    before: str
    after: str
    unified_diff: str
    node_name: str | None = None
    where: str = ""


# --- diff helpers -------------------------------------------------------------

def _lines(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines() or [""]:
        out.extend(_SENTENCE_RE.split(line))
    return out


def unified_diff(before: Any, after: Any) -> str:
    a = before if isinstance(before, str) else json.dumps(before, ensure_ascii=False, indent=2, sort_keys=True)
    b = after if isinstance(after, str) else json.dumps(after, ensure_ascii=False, indent=2, sort_keys=True)
    return "\n".join(difflib.unified_diff(_lines(a), _lines(b), "input", "output", lineterm="", n=2))


def char_diff(before: str, after: str) -> str:
    """Inline character-level diff: [-removed-]{+added+}. Tiny equal runs
    between edits are folded in so a changed word reads as one edit."""
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    segs: list[list] = []  # [equal?, a_text, b_text]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg = [tag == "equal", before[i1:i2], after[j1:j2]]
        if segs and not seg[0] and not segs[-1][0]:
            segs[-1][1] += seg[1]
            segs[-1][2] += seg[2]
        else:
            segs.append(seg)
    merged: list[list] = []
    for i, seg in enumerate(segs):
        between_edits = 0 < i < len(segs) - 1 and seg[0] and len(seg[1]) < 3 and not seg[1].isspace()
        if between_edits:
            seg = [False, seg[1], seg[2]]
        if merged and not seg[0] and not merged[-1][0]:
            merged[-1][1] += seg[1]
            merged[-1][2] += seg[2]
        else:
            merged.append(list(seg))
    # Widen edits to whole words: "agre|es" -> "agrees".
    for k, seg in enumerate(merged):
        if seg[0]:
            continue
        if k > 0 and merged[k - 1][0]:
            m = re.search(r"\w+$", merged[k - 1][1])
            if m and re.match(r"\w", seg[1] or seg[2]):
                merged[k - 1][1] = merged[k - 1][2] = merged[k - 1][1][: m.start()]
                seg[1], seg[2] = m.group(0) + seg[1], m.group(0) + seg[2]
        if k + 1 < len(merged) and merged[k + 1][0]:
            m = re.match(r"\w+", merged[k + 1][1])
            if m and re.search(r"\w$", seg[1] or seg[2]):
                merged[k + 1][1] = merged[k + 1][2] = merged[k + 1][1][m.end():]
                seg[1], seg[2] = seg[1] + m.group(0), seg[2] + m.group(0)
    out = []
    for equal, a, b in merged:
        if equal:
            out.append(a)
        else:
            out.append((f"[-{a}-]" if a else "") + (f"{{+{b}+}}" if b else ""))
    return "".join(out)


def _describe_tokens(changes: list, side: str) -> str:
    """"currency ₹1008: 1 of 9 lost (9 -> 8); ... Context in the input: «...»"."""
    verb = "lost" if side == "input" else "added"
    parts = []
    for t in changes:
        if verb == "lost":
            count = "all lost" if t.after == 0 else f"{t.delta} of {t.before} lost"
        else:
            count = "new" if t.before == 0 else f"{t.delta} more"
        parts.append(f"{t.kind} {t.token}: {count} ({t.before} -> {t.after})")
    contexts, seen = [], []
    for t in changes:
        for pos, c in zip(t.positions, t.contexts):
            if all(abs(pos - p) > 15 for p in seen):  # same spot, e.g. "₹1,008" and "1008"
                seen.append(pos)
                contexts.append(c)
    text = "; ".join(parts) + "."
    if contexts:
        text += f" Context in the {side}: " + " | ".join(f"«{c}»" for c in contexts[:3])
    return text


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _clip(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    return value


class PairChecker:
    def __init__(self, inp: Flow, out: Flow, mapping: Mapping, policy: Policy):
        self.inp, self.out, self.m, self.policy = inp, out, mapping, policy
        self.findings: list[Finding] = []
        self.text_changes: list[TextChange] = []
        self.position_changes: list[str] = []
        self.key_order_paths: list[str] = []
        self.topology_equivalent: bool | None = None
        self.changed_output_edges: set[str] = set()

    # -- emit ----------------------------------------------------------------------
    def emit(self, category: str, message: str, node_id: str | None = None, path: str = "",
             before: Any = None, after: Any = None, diff: str | None = None, **extra) -> Finding:
        f = Finding(self.policy.severity(category), category, message, node_id=node_id, path=path,
                    before=_clip(before), after=_clip(after), unified_diff=diff, extra=extra)
        self.findings.append(f)
        return f

    def _name(self, node_id: Any) -> str:
        node = self.inp.node_by_id.get(node_id) if isinstance(node_id, str) else None
        return f"'{node.get('name')}'" if node and node.get("name") else f"'{node_id}'"

    def where(self, node_id: str | None, path: str) -> str:
        """Readable location: "Node 'Greeting', field onEnterSay.message"."""
        if not node_id:
            return f"Field {path}"
        rest = path.split("]", 1)[1].lstrip(".") if path.startswith("nodes[") else path
        return f"Node {self._name(node_id)}, field {rest}"

    def _out_name(self, node_id: Any) -> str:
        node = self.out.node_by_id.get(node_id) if isinstance(node_id, str) else None
        return f"'{node.get('name')}'" if node and node.get("name") else f"'{node_id}'"

    # -- run -----------------------------------------------------------------------
    def run(self) -> "PairChecker":
        self.check_envelope()
        self.check_data()
        self.check_conv()
        for match in self.m.matches:
            if match.input_id and match.output_id:
                self.check_node(self.inp.node_by_id[match.input_id], self.out.node_by_id[match.output_id])
        self.check_topology()
        if self.position_changes:
            self.emit("position_changed",
                      f"Canvas positions changed for {len(self.position_changes)} item(s): "
                      f"{', '.join(self.position_changes)}. Layout only; no effect on the conversation.",
                      path="position")
        if self.key_order_paths:
            sample = ", ".join(self.key_order_paths[:5]) + (" ..." if len(self.key_order_paths) > 5 else "")
            self.emit("key_order_changed",
                      f"JSON key order differs in {len(self.key_order_paths)} object(s) ({sample}). No effect.",
                      path="(key order)")
        return self

    # -- generic -------------------------------------------------------------------
    def keys_diff(self, path: str, a: dict, b: dict, node_id: str | None) -> None:
        for key in a:
            if key not in b:
                self.emit("schema_key_removed", f"Key '{key}' was removed at {path or 'top level'}.",
                          node_id, f"{path}.{key}".lstrip("."), before=a[key])
        for key in b:
            if key not in a:
                self.emit("schema_key_added", f"Unknown key '{key}' was added at {path or 'top level'}.",
                          node_id, f"{path}.{key}".lstrip("."), after=b[key])
        common_a = [k for k in a if k in b]
        common_b = [k for k in b if k in a]
        if common_a != common_b:
            self.key_order_paths.append(path or "(root)")

    def generic(self, path: str, a: Any, b: Any, node_id: str | None, text_zone: bool = False) -> None:
        """Compare anything no dedicated check owns: schema drift, type drift, value changes."""
        if a == b and _kind(a) == _kind(b):
            if isinstance(a, dict) and list(a) != list(b):
                self.key_order_paths.append(path)
            return
        ka, kb = _kind(a), _kind(b)
        if ka != kb and "null" not in (ka, kb):
            self.emit("type_drift", f"{path} changed type from {ka} to {kb} ({a!r} -> {b!r}).",
                      node_id, path, before=a, after=b)
            return
        if isinstance(a, dict) and isinstance(b, dict):
            self.keys_diff(path, a, b, node_id)
            for key in a:
                if key in b:
                    self.generic(f"{path}.{key}", a[key], b[key], node_id,
                                 text_zone or key in ("prompts", "overlays"))
            return
        if isinstance(a, list) and isinstance(b, list):
            key = self._list_key(a, b)
            if key:
                ia = {item[key]: item for item in a}
                ib = {item[key]: item for item in b}
                for k, item in ia.items():
                    if k not in ib:
                        self.emit("field_changed", f"{path}: item '{k}' was removed.", node_id, f"{path}[{k}]", before=item)
                    else:
                        self.generic(f"{path}[{k}]", item, ib[k], node_id, text_zone)
                for k, item in ib.items():
                    if k not in ia:
                        self.emit("field_changed", f"{path}: item '{k}' was added.", node_id, f"{path}[{k}]", after=item)
                if [k for k in ia if k in ib] != [k for k in ib if k in ia]:
                    self.emit("field_changed", f"{path}: items were reordered.", node_id, path,
                              before=list(ia), after=list(ib))
                return
            if len(a) == len(b):
                for i, (x, y) in enumerate(zip(a, b)):
                    self.generic(f"{path}[{i}]", x, y, node_id, text_zone)
                return
            self.emit("field_changed", f"{path} changed (list of {len(a)} -> {len(b)} items).", node_id, path,
                      before=a, after=b, diff=unified_diff(a, b))
            return
        if isinstance(a, str) and isinstance(b, str):
            last = re.sub(r"\[.*?\]$", "", path.rsplit(".", 1)[-1])
            if text_zone or last in TEXT_KEYS:
                self.text(path, a, b, node_id)
                return
        self.emit("field_changed", f"{path} changed from {a!r} to {b!r}.", node_id, path, before=a, after=b)

    @staticmethod
    def _list_key(a: list, b: list) -> str | None:
        for key in ("id", "name"):
            ok = True
            for items in (a, b):
                vals = [x.get(key) if isinstance(x, dict) else None for x in items]
                if not items or any(not isinstance(v, str) for v in vals) or len(set(vals)) != len(vals):
                    ok = False
                    break
            if ok:
                return key
        return None

    # -- text ----------------------------------------------------------------------
    def text(self, path: str, before: Any, after: Any, node_id: str | None) -> None:
        a = before if isinstance(before, str) else ""
        b = after if isinstance(after, str) else ""
        if a == b:
            return
        where = self.where(node_id, path)

        lost, added = token_delta(a, b)
        if lost:
            self.emit("token_lost", f"{where} lost tokens that must be preserved: {_describe_tokens(lost, 'input')}",
                      node_id, path, before=a, after=b, diff=unified_diff(a, b),
                      tokens=[t.to_dict() for t in lost])
        if added:
            self.emit("token_added", f"{where} gained tokens not in the input: {_describe_tokens(added, 'output')}",
                      node_id, path, before=a, after=b, diff=unified_diff(a, b),
                      tokens=[t.to_dict() for t in added])

        na, nb = normalize_text(a), normalize_text(b)
        if na == nb:
            if collapse_ws(a) == collapse_ws(b):
                self.emit("text_whitespace_only", f"{where}: whitespace-only change.", node_id, path, before=a, after=b)
            else:
                self.emit("text_normalization_only",
                          f"{where}: only Unicode normalization or markdown escaping changed; the text is the same.",
                          node_id, path, before=a, after=b)
            return
        reasons = corruption_reasons(a, b)
        if reasons:
            self.emit("text_corrupted", f"{where} is corrupted: {'; '.join(reasons)}.",
                      node_id, path, before=a, after=b, diff=unified_diff(a, b))
            return
        if na.startswith(nb):
            self.emit("text_truncated",
                      f"{where} was truncated: output keeps only the first {len(nb)} of {len(na)} characters.",
                      node_id, path, before=a, after=b, diff=unified_diff(a, b))
            return
        if (lost or added) and strip_tokens(a) == strip_tokens(b):
            return  # the token findings above fully explain this change
        node = self.inp.node_by_id.get(node_id) if node_id else None
        self.text_changes.append(TextChange(
            group=node_id or "__global__", node_id=node_id, path=path, before=a, after=b,
            unified_diff=unified_diff(a, b), node_name=(node or {}).get("name"), where=where,
        ))

    # -- envelope / data / conversationNodes ------------------------------------
    def check_envelope(self) -> None:
        if self.inp.envelope != self.out.envelope:
            self.emit("envelope_changed",
                      f"Document wrapper changed from '{self.inp.envelope}' to '{self.out.envelope}' form.",
                      path="(envelope)", before=self.inp.envelope, after=self.out.envelope)
            return
        if self.inp.envelope != "envelope":
            return
        a, b = self.inp.raw, self.out.raw
        self.keys_diff("", a, b, None)
        if a.get("exportedAt") != b.get("exportedAt") and "exportedAt" in a and "exportedAt" in b:
            self.emit("exported_at_changed", "exportedAt timestamp changed (expected on every export).",
                      path="exportedAt", before=a.get("exportedAt"), after=b.get("exportedAt"))
        for key in a:
            if key in b and key not in ENVELOPE_HANDLED:
                self.generic(key, a[key], b[key], None)

    def check_data(self) -> None:
        a, b = self.inp.data, self.out.data
        if self.inp.envelope != "conversation_nodes" and self.out.envelope != "conversation_nodes":
            self.keys_diff("data", a, b, None)
        for key in ("prompt", "initialMessage"):
            if key in a or key in b:
                self.text(f"data.{key}", a.get(key), b.get(key), None)
        self.check_prompts(a.get("prompts"), b.get("prompts"))
        if "supportedModes" in a or "supportedModes" in b:
            if a.get("supportedModes") != b.get("supportedModes"):
                self.emit("supported_modes_changed", "supportedModes changed.", path="data.supportedModes",
                          before=a.get("supportedModes"), after=b.get("supportedModes"))
        for key in a:
            if key in b and key not in DATA_HANDLED:
                self.generic(f"data.{key}", a[key], b[key], None)

    def check_prompts(self, a: Any, b: Any) -> None:
        if a is None and b is None:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                if key not in b:
                    self.emit("prompt_key_removed", f"Prompt '{key}' was removed from data.prompts.",
                              path=f"data.prompts.{key}", before=a[key])
            for key in b:
                if key not in a:
                    self.emit("prompt_key_added", f"Prompt '{key}' was added to data.prompts.",
                              path=f"data.prompts.{key}", after=b[key])
            for key in a:
                if key in b:
                    if isinstance(a[key], str) and isinstance(b[key], str):
                        self.text(f"data.prompts.{key}", a[key], b[key], None)
                    else:
                        self.generic(f"data.prompts.{key}", a[key], b[key], None, text_zone=True)
            return
        self.generic("data.prompts", a, b, None, text_zone=True)

    def check_conv(self) -> None:
        a, b = self.inp.conv, self.out.conv
        self.keys_diff("conversationNodes", a, b, None)

        if self.inp.nodes_container != self.out.nodes_container:
            self.emit("nodes_container_changed",
                      f"conversationNodes.nodes changed shape from a {self.inp.nodes_container} to a "
                      f"{self.out.nodes_container}. Some tools may not read the new shape.",
                      path="conversationNodes.nodes", before=self.inp.nodes_container, after=self.out.nodes_container)

        # Start node (B3).
        start_in, start_out = self.inp.start_node_id, self.out.start_node_id
        if self.m.map_id(start_in) != start_out:
            self.emit("start_node_changed",
                      f"Start node changed: the input starts at {self._name(start_in)} ({start_in}), "
                      f"the output starts at {self._out_name(start_out)} ({start_out}).",
                      path="conversationNodes.startNodeId", before=start_in, after=start_out)

        # Node array order (B11).
        in_order = [self.m.forward[i] for i in self.inp.node_by_id if i in self.m.forward]
        out_order = [o for o in self.out.node_by_id if o in set(in_order)]
        if in_order != out_order:
            self.emit("node_reordered", "Nodes appear in a different order in the array. Order has no effect on routing.",
                      path="conversationNodes.nodes",
                      before=[self.m.backward.get(o, o) for o in in_order], after=[self.m.backward.get(o, o) for o in out_order])

        if a.get("startNodePosition") != b.get("startNodePosition") and "startNodePosition" in a and "startNodePosition" in b:
            self.position_changes.append("startNodePosition")

        self.check_global_variables(self.inp.global_variables, self.out.global_variables)
        for key in a:
            if key in b and key not in CONV_HANDLED:
                self.generic(f"conversationNodes.{key}", a[key], b[key], None)

    def check_global_variables(self, a: list, b: list) -> None:
        ia = {v.get("name"): v for v in a if isinstance(v, dict)}
        ib = {v.get("name"): v for v in b if isinstance(v, dict)}
        removed = [n for n in ia if n not in ib]
        added = [n for n in ib if n not in ia]
        if removed or added:
            parts = []
            if removed:
                parts.append(f"removed: {', '.join(map(str, removed))}")
            if added:
                parts.append(f"added: {', '.join(map(str, added))}")
            self.emit("global_variable_changed",
                      f"Global variables changed ({'; '.join(parts)}). Placeholders that use these names will break.",
                      path="globalVariables", before=sorted(map(str, ia)), after=sorted(map(str, ib)))
        for name, va in ia.items():
            vb = ib.get(name)
            if vb is None:
                continue
            path = f"globalVariables[{name}]"
            if va.get("type") != vb.get("type"):
                self.emit("global_variable_changed",
                          f"Global variable '{name}' changed type from {va.get('type')!r} to {vb.get('type')!r}.",
                          path=f"{path}.type", before=va.get("type"), after=vb.get("type"))
            self.keys_diff(path, va, vb, None)
            for key in va:
                if key in vb and key not in ("name", "type"):
                    self.generic(f"{path}.{key}", va[key], vb[key], None)

    # -- nodes ----------------------------------------------------------------------
    def check_node(self, a: dict, b: dict) -> None:
        nid = a["id"]
        base = f"nodes[{nid}]"
        self.keys_diff(base, a, b, nid)

        if a.get("type") != b.get("type"):
            self.emit("node_type_changed",
                      f"Node {self._name(nid)} changed type from {a.get('type')!r} to {b.get('type')!r}.",
                      nid, f"{base}.type", before=a.get("type"), after=b.get("type"))
        if a.get("name") != b.get("name"):
            self.emit("node_name_changed", f"Node {self._name(nid)} was renamed to {b.get('name')!r}.",
                      nid, f"{base}.name", before=a.get("name"), after=b.get("name"))
        if "position" in a and "position" in b and a["position"] != b["position"]:
            self.position_changes.append(str(a.get("name") or nid))

        # Protected (B7).
        ga, gb = a.get("globalNode"), b.get("globalNode")
        ea = ga.get("enabled") if isinstance(ga, dict) else None
        eb = gb.get("enabled") if isinstance(gb, dict) else None
        if ea != eb:
            self.emit("global_node_changed", f"Node {self._name(nid)}: globalNode.enabled changed from {ea!r} to {eb!r}.",
                      nid, f"{base}.globalNode.enabled", before=ea, after=eb)
        if isinstance(ga, dict) and isinstance(gb, dict):
            self.keys_diff(f"{base}.globalNode", ga, gb, nid)
            for key in ga:
                if key in gb and key != "enabled":
                    self.generic(f"{base}.globalNode.{key}", ga[key], gb[key], nid)
        if a.get("allowGoBack") != b.get("allowGoBack"):
            self.emit("allow_go_back_changed",
                      f"Node {self._name(nid)}: allowGoBack changed from {a.get('allowGoBack')!r} to {b.get('allowGoBack')!r}.",
                      nid, f"{base}.allowGoBack", before=a.get("allowGoBack"), after=b.get("allowGoBack"))
        for group, category in (("goBackEdges", "go_back_edges_changed"), ("jumpToEdges", "jump_to_edges_changed")):
            ca, cb = self._canonical_edges(a, group, mapped=True), self._canonical_edges(b, group, mapped=False)
            if ca != cb:
                self.emit(category, f"Node {self._name(nid)}: {group} changed.", nid, f"{base}.{group}",
                          before=a.get(group), after=b.get(group), diff=unified_diff(ca, cb))

        # Text.
        if "instruction" in a or "instruction" in b:
            self.text(f"{base}.instruction", a.get("instruction"), b.get("instruction"), nid)
        sa, sb = a.get("onEnterSay"), b.get("onEnterSay")
        if isinstance(sa, dict) or isinstance(sb, dict):
            sa, sb = sa if isinstance(sa, dict) else {}, sb if isinstance(sb, dict) else {}
            if "message" in sa or "message" in sb:
                self.text(f"{base}.onEnterSay.message", sa.get("message"), sb.get("message"), nid)
            self.keys_diff(f"{base}.onEnterSay", sa, sb, nid)
            for key in sa:
                if key in sb and key != "message":
                    self.generic(f"{base}.onEnterSay.{key}", sa[key], sb[key], nid)
        else:
            self.generic(f"{base}.onEnterSay", sa, sb, nid, text_zone=True)

        self.check_overlays(f"{base}.overlays", a.get("overlays"), b.get("overlays"), nid)
        if a.get("type") == "switch" or b.get("type") == "switch":
            self.check_switch(a, b)
        self.check_edges(a, b)

        switch_keys = {"switchExpression", "switchDefaultEdgeId", "switchCases"}
        for key in a:
            if key in b and key not in NODE_HANDLED:
                self.generic(f"{base}.{key}", a[key], b[key], nid)
            elif key in b and key in switch_keys and "switch" not in (a.get("type"), b.get("type")):
                self.generic(f"{base}.{key}", a[key], b[key], nid)

    def _canonical_edges(self, node: dict, group: str, mapped: bool) -> list:
        out = []
        for edge in edges_of(node, group):
            e = {k: v for k, v in edge.items() if k != "id"}
            if "destinationNodeId" in e and mapped:
                e["destinationNodeId"] = self.m.map_id(e["destinationNodeId"])
            if isinstance(e.get("transitionCondition"), dict):
                e["transitionCondition"] = dict(e["transitionCondition"],
                                                prompt=normalize_text(condition_text(edge)))
            out.append(e)
        return out

    def check_overlays(self, path: str, a: Any, b: Any, node_id: str | None) -> None:
        if a is None and b is None:
            return
        if not (isinstance(a, dict) and isinstance(b, dict)):
            self.generic(path, a, b, node_id, text_zone=True)
            return
        for key in a:
            if key not in b:
                category = "overlay_language_removed" if key.startswith("lang:") else "schema_key_removed"
                self.emit(category, f"{self._name(node_id) if node_id else ''} lost its '{key}' translation at {path}.".strip(),
                          node_id, f"{path}.{key}", before=a[key])
        for key in b:
            if key not in a:
                category = "overlay_language_added" if key.startswith("lang:") else "schema_key_added"
                self.emit(category, f"{self._name(node_id) if node_id else ''} gained a '{key}' overlay at {path}.".strip(),
                          node_id, f"{path}.{key}", after=b[key])
        for key in a:
            if key in b:
                self.generic(f"{path}.{key}", a[key], b[key], node_id, text_zone=True)

    # -- switch (B6) ---------------------------------------------------------------
    def check_switch(self, a: dict, b: dict) -> None:
        nid = a["id"]
        base = f"nodes[{nid}]"
        if a.get("switchExpression") != b.get("switchExpression"):
            self.emit("switch_expression_changed",
                      f"Switch node {self._name(nid)}: switchExpression changed.", nid, f"{base}.switchExpression",
                      before=a.get("switchExpression"), after=b.get("switchExpression"))

        def dest(node: dict, edge_id: Any, mapped: bool) -> Any:
            for edge in edges_of(node):
                if edge.get("id") == edge_id:
                    d = edge.get("destinationNodeId")
                    return self.m.map_id(d) if mapped else d
            return f"<no edge {edge_id}>"

        ca = a.get("switchCases") if isinstance(a.get("switchCases"), dict) else {}
        cb = b.get("switchCases") if isinstance(b.get("switchCases"), dict) else {}
        for case in sorted(set(ca) | set(cb), key=str):
            if case not in cb:
                self.emit("switch_case_changed", f"Switch node {self._name(nid)}: case '{case}' was removed.",
                          nid, f"{base}.switchCases.{case}", before=ca[case])
            elif case not in ca:
                self.emit("switch_case_changed", f"Switch node {self._name(nid)}: case '{case}' was added.",
                          nid, f"{base}.switchCases.{case}", after=cb[case])
            else:
                da, db = dest(a, ca[case], True), dest(b, cb[case], False)
                if da != db:
                    self.changed_output_edges.add(str(cb[case]))
                    self.emit("switch_case_changed",
                              f"Switch node {self._name(nid)}: case '{case}' now routes to {self._out_name(db)} "
                              f"instead of {self._out_name(da)}.",
                              nid, f"{base}.switchCases.{case}", before=ca[case], after=cb[case])
        if "switchDefaultEdgeId" in a or "switchDefaultEdgeId" in b:
            da = dest(a, a.get("switchDefaultEdgeId"), True)
            db = dest(b, b.get("switchDefaultEdgeId"), False)
            if da != db:
                self.changed_output_edges.add(str(b.get("switchDefaultEdgeId")))
                self.emit("switch_default_changed",
                          f"Switch node {self._name(nid)}: the default route now goes to {self._out_name(db)} "
                          f"instead of {self._out_name(da)}.",
                          nid, f"{base}.switchDefaultEdgeId",
                          before=a.get("switchDefaultEdgeId"), after=b.get("switchDefaultEdgeId"))

    # -- edges (B4) ----------------------------------------------------------------
    def match_edges(self, a: dict, b: dict) -> list[tuple[int, int]]:
        ea, eb = edges_of(a), edges_of(b)
        free_a, free_b = set(range(len(ea))), set(range(len(eb)))
        pairs: list[tuple[int, int]] = []

        def take(i: int, j: int) -> None:
            pairs.append((i, j))
            free_a.discard(i)
            free_b.discard(j)

        ids_b = {}
        for j, e in enumerate(eb):
            ids_b.setdefault(e.get("id"), j)
        for i, e in enumerate(ea):
            j = ids_b.get(e.get("id"))
            if e.get("id") is not None and j in free_b:
                take(i, j)

        def by_sig(edges, free, node):
            sigs: dict[str, list[int]] = {}
            for k in free:
                sig = edge_signature(node, edges[k])
                if normalize_text(condition_text(edges[k])) or node.get("type") == "switch":
                    sigs.setdefault(sig, []).append(k)
            return sigs
        sa, sb = by_sig(ea, free_a, a), by_sig(eb, free_b, b)
        for sig, lst in sa.items():
            if len(lst) == 1 and len(sb.get(sig, [])) == 1:
                take(lst[0], sb[sig][0])

        for i in sorted(free_a):
            for j in sorted(free_b):
                if j == i and self.m.map_id(ea[i].get("destinationNodeId")) == eb[j].get("destinationNodeId"):
                    take(i, j)
                    break
        for i in sorted(free_a):
            for j in sorted(free_b):
                if self.m.map_id(ea[i].get("destinationNodeId")) == eb[j].get("destinationNodeId"):
                    take(i, j)
                    break
        return sorted(pairs)

    def check_edges(self, a: dict, b: dict) -> None:
        nid = a["id"]
        base = f"nodes[{nid}].edges"
        ea, eb = edges_of(a), edges_of(b)
        pairs = self.match_edges(a, b)
        matched_a = {i for i, _ in pairs}
        matched_b = {j for _, j in pairs}

        for i, e in enumerate(ea):
            if i not in matched_a:
                self.emit("edge_removed",
                          f"Node {self._name(nid)}: the transition to {self._name(e.get('destinationNodeId'))} "
                          f"(\"{condition_text(e) or 'no condition'}\") is missing from the output.",
                          nid, f"{base}[{e.get('id')}]", before=e)
        for j, e in enumerate(eb):
            if j not in matched_b:
                self.changed_output_edges.add(str(e.get("id")))
                self.emit("edge_added",
                          f"Node {self._name(nid)}: output has a new transition to {self._out_name(e.get('destinationNodeId'))} "
                          f"(\"{condition_text(e) or 'no condition'}\").",
                          nid, f"{base}[{e.get('id')}]", after=e)

        is_switch = a.get("type") == "switch"
        for i, j in pairs:
            x, y = ea[i], eb[j]
            path = f"{base}[{x.get('id')}]"
            self.keys_diff(path, x, y, nid)
            if x.get("id") != y.get("id"):
                self.emit("edge_id_changed", f"Node {self._name(nid)}: edge id changed from '{x.get('id')}' to '{y.get('id')}'.",
                          nid, f"{path}.id", before=x.get("id"), after=y.get("id"))
            want, got = self.m.map_id(x.get("destinationNodeId")), y.get("destinationNodeId")
            if want != got:
                self.changed_output_edges.add(str(y.get("id")))
                self.emit("edge_redirected",
                          f"Node {self._name(nid)}: edge '{x.get('id')}' (\"{condition_text(x) or 'no condition'}\") "
                          f"should lead to {self._out_name(want)} but now leads to {self._out_name(got)}.",
                          nid, f"{path}.destinationNodeId", before=x.get("destinationNodeId"), after=got)
            self.check_condition(path, x, y, nid, is_switch)
            if x.get("goBackToPrevious") != y.get("goBackToPrevious"):
                self.emit("go_back_to_previous_changed",
                          f"Node {self._name(nid)}: edge '{x.get('id')}' goBackToPrevious changed.",
                          nid, f"{path}.goBackToPrevious", before=x.get("goBackToPrevious"), after=y.get("goBackToPrevious"))
            if "transitionMessage" in x or "transitionMessage" in y:
                tx, ty = x.get("transitionMessage"), y.get("transitionMessage")
                if isinstance(tx, (str, type(None))) and isinstance(ty, (str, type(None))):
                    self.text(f"{path}.transitionMessage", tx, ty, nid)
                else:
                    self.generic(f"{path}.transitionMessage", tx, ty, nid, text_zone=True)
            self.check_overlays(f"{path}.overlays", x.get("overlays"), y.get("overlays"), nid)
            for key in x:
                if key in y and key not in EDGE_HANDLED:
                    self.generic(f"{path}.{key}", x[key], y[key], nid)

        order = [j for _, j in sorted(pairs)]
        if order != sorted(order):
            self.emit("edge_order_changed",
                      f"Node {self._name(nid)}: transitions are listed in a different order. "
                      "Order can decide which transition wins when several match.",
                      nid, base, before=[ea[i].get("id") for i, _ in sorted(pairs)],
                      after=[eb[j].get("id") for j in sorted(order)])

    def check_condition(self, path: str, x: dict, y: dict, nid: str, is_switch: bool) -> None:
        tx, ty = x.get("transitionCondition"), y.get("transitionCondition")
        if isinstance(tx, dict) and isinstance(ty, dict):
            self.keys_diff(f"{path}.transitionCondition", tx, ty, nid)
            for key in tx:
                if key in ty and key != "prompt":
                    self.generic(f"{path}.transitionCondition.{key}", tx[key], ty[key], nid)
        elif tx != ty and not (isinstance(tx, dict) or isinstance(ty, dict)):
            self.generic(f"{path}.transitionCondition", tx, ty, nid)
        ca, cb = condition_text(x), condition_text(y)
        cpath = f"{path}.transitionCondition.prompt"
        if ca == cb:
            return
        if normalize_text(ca) == normalize_text(cb):
            self.emit("condition_whitespace_only",
                      f"Node {self._name(nid)}: condition of edge '{x.get('id')}' changed only in whitespace or Unicode form.",
                      nid, cpath, before=ca, after=cb)
            return
        self.changed_output_edges.add(str(y.get("id")))
        if ca.strip() and not cb.strip():
            self.emit("condition_emptied",
                      f"Node {self._name(nid)}: the condition of edge '{x.get('id')}' was erased. "
                      "The agent no longer knows when to take this transition.",
                      nid, cpath, before=ca, after=cb)
            return
        self.emit("condition_changed",
                  f"Node {self._name(nid)}: the condition of edge '{x.get('id')}' was reworded. "
                  "Pre-written transition conditions must stay exactly as written.",
                  nid, cpath, before=ca, after=cb, diff=char_diff(ca, cb))

    # -- topology (B5) ---------------------------------------------------------------
    def edge_multiset(self, flow: Flow, mapped: bool) -> Counter:
        result: Counter = Counter()
        for node in flow.nodes:
            src = node.get("id")
            for edge in edges_of(node):
                dst = edge_destination(node, edge)
                if mapped:
                    src_m, dst_m = self.m.map_id(src), self.m.map_id(dst)
                else:
                    src_m, dst_m = src, dst
                result[(str(src_m), str(dst_m), signature_hash(edge_signature(node, edge)))] += 1
        return result

    def check_topology(self) -> None:
        edges_in, edges_out = self.edge_multiset(self.inp, True), self.edge_multiset(self.out, False)
        reach_in = {str(self.m.map_id(n)) for n in reachable_from_start(self.inp)}
        reach_out = set(reachable_from_start(self.out))
        term_in = {str(self.m.map_id(n["id"])) for n in self.inp.nodes if n.get("type") == "end" and "id" in n}
        term_out = {n["id"] for n in self.out.nodes if n.get("type") == "end" and "id" in n}

        problems = []
        if edges_in != edges_out:
            missing = sum((edges_in - edges_out).values())
            extra = sum((edges_out - edges_in).values())
            problems.append(f"{missing} transition(s) from the input have no exact counterpart in the output "
                            f"and {extra} output transition(s) are new or different")
        if reach_in != reach_out:
            problems.append("the set of nodes reachable from the start node differs")
        if term_in != term_out:
            problems.append("the set of end nodes differs")
        self.topology_equivalent = not problems
        if problems:
            self.emit("topology_not_equivalent",
                      "The conversation graph is not equivalent to the input: " + "; ".join(problems) + ".",
                      path="(graph)", topology_equivalent=False)
        elif not self.m.is_identity:
            self.emit("topology_equivalent",
                      "Despite the id changes, the conversation graph is equivalent to the input "
                      "(same transitions, same reachable nodes, same end nodes).",
                      path="(graph)", topology_equivalent=True)
