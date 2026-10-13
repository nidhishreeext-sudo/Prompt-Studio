"""B1. Node matching. Every other pair check runs on the mapping produced here."""

from __future__ import annotations

import difflib
import hashlib
from collections import Counter
from dataclasses import dataclass, field

from .loader import Flow, condition_text, edges_of
from .model import Finding, NodeMatch
from .policy import SCORE_WEIGHTS, Policy
from .tokens import normalize_text


def edge_signature(node: dict, edge: dict) -> str:
    """What an edge *means*, independent of ids: its condition text, plus for
    switch nodes the case values (and default flag) that route to it."""
    sig = normalize_text(condition_text(edge))
    if node.get("type") == "switch":
        cases = node.get("switchCases") if isinstance(node.get("switchCases"), dict) else {}
        keys = sorted(str(k) for k, v in cases.items() if v == edge.get("id"))
        default = node.get("switchDefaultEdgeId") == edge.get("id")
        sig += f"\x00cases={','.join(keys)}\x00default={default}"
    return sig


def signature_hash(sig: str) -> str:
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def text_similarity(a: str | None, b: str | None) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _on_enter_say(node: dict) -> str:
    say = node.get("onEnterSay")
    return say.get("message") or "" if isinstance(say, dict) else ""


def node_similarity(a: dict, b: dict) -> float:
    ca = Counter(signature_hash(edge_signature(a, e)) for e in edges_of(a))
    cb = Counter(signature_hash(edge_signature(b, e)) for e in edges_of(b))
    if not ca and not cb:
        edge_overlap = 1.0
    else:
        edge_overlap = sum((ca & cb).values()) / sum((ca | cb).values())
    w = SCORE_WEIGHTS
    return (
        w["edges"] * edge_overlap
        + w["instruction"] * text_similarity(a.get("instruction"), b.get("instruction"))
        + w["name"] * text_similarity(a.get("name"), b.get("name"))
        + w["on_enter_say"] * text_similarity(_on_enter_say(a), _on_enter_say(b))
        + w["type"] * (1.0 if a.get("type") == b.get("type") else 0.0)
    )


@dataclass
class Mapping:
    forward: dict[str, str] = field(default_factory=dict)     # input id -> output id
    matches: list[NodeMatch] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    id_shift: bool = False

    @property
    def backward(self) -> dict[str, str]:
        return {v: k for k, v in self.forward.items()}

    def map_id(self, input_id: object) -> object:
        """Translate an input node id into output-id space. Unmatched ids are
        tagged so they can never accidentally equal a real output id."""
        if isinstance(input_id, str) and input_id in self.forward:
            return self.forward[input_id]
        return f"<unmatched:{input_id}>"

    @property
    def is_identity(self) -> bool:
        return all(k == v for k, v in self.forward.items()) and not self.removed and not self.added


def _greedy(pairs: list[tuple[float, str, str]], threshold: float, taken_in: set, taken_out: set) -> list[tuple[str, str, float]]:
    result = []
    for score, in_id, out_id in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if score < threshold or in_id in taken_in or out_id in taken_out:
            continue
        taken_in.add(in_id)
        taken_out.add(out_id)
        result.append((in_id, out_id, score))
    return result


def _content_pairs(inp: Flow, out: Flow, in_ids, out_ids) -> list[tuple[float, str, str]]:
    pairs = []
    for in_id in in_ids:
        a = inp.node_by_id[in_id]
        for out_id in out_ids:
            b = out.node_by_id[out_id]
            if a.get("type") == b.get("type"):
                pairs.append((node_similarity(a, b), in_id, out_id))
    return pairs


def match_nodes(inp: Flow, out: Flow, policy: Policy) -> Mapping:
    low, high = policy.threshold("id_shift_low"), policy.threshold("id_shift_high")
    threshold = policy.threshold("match_threshold")
    in_ids = list(inp.node_by_id)
    out_ids = list(out.node_by_id)
    m = Mapping()

    # Pass 1: by id, scored.
    id_scores = {i: node_similarity(inp.node_by_id[i], out.node_by_id[i]) for i in in_ids if i in out.node_by_id}

    # Id-shift detection: a weak id match while another output node is a strong content match.
    shifted = {}
    for in_id, score in id_scores.items():
        if score >= low:
            continue
        a = inp.node_by_id[in_id]
        best = max(((node_similarity(a, out.node_by_id[o]), o) for o in out_ids if o != in_id), default=(0.0, None))
        if best[1] is not None and best[0] >= high:
            shifted[in_id] = best

    taken_in: set[str] = set()
    taken_out: set[str] = set()
    if shifted:
        m.id_shift = True
        content = _greedy(_content_pairs(inp, out, in_ids, out_ids), threshold, taken_in, taken_out)
        for in_id, out_id, score in content:
            m.forward[in_id] = out_id
            method = "id" if in_id == out_id else "id_shift"
            m.matches.append(NodeMatch(in_id, out_id, method=method, score=score))
        moved = sorted((i, o) for i, o in m.forward.items() if i != o)
        lines = [f"'{inp.node_by_id[i].get('name')}' was {i}, now {o}" for i, o in moved]
        m.findings.append(Finding(
            policy.severity("node_id_shift"), "node_id_shift",
            "Node ids were shifted: node content now sits under a different node's id, so any reference "
            "by id now points at the wrong node. Content-based mapping: " + "; ".join(lines) + ".",
            path="conversationNodes.nodes",
            before={i: inp.node_by_id[i].get("name") for i, _ in moved},
            after={o: out.node_by_id[o].get("name") for _, o in moved},
            extra={"mapping": dict(moved)},
        ))
    else:
        for in_id, score in id_scores.items():
            m.forward[in_id] = in_id
            taken_in.add(in_id)
            taken_out.add(in_id)
            m.matches.append(NodeMatch(in_id, in_id, method="id", score=score))
        # Pass 2: leftovers by content.
        rest_in = [i for i in in_ids if i not in taken_in]
        rest_out = [o for o in out_ids if o not in taken_out]
        for in_id, out_id, score in _greedy(_content_pairs(inp, out, rest_in, rest_out), threshold, taken_in, taken_out):
            m.forward[in_id] = out_id
            m.matches.append(NodeMatch(in_id, out_id, method="content", score=score))
            name = inp.node_by_id[in_id].get("name")
            m.findings.append(Finding(
                policy.severity("node_id_changed"), "node_id_changed",
                f"Node '{name}' kept its content but its id changed from '{in_id}' to '{out_id}'.",
                node_id=in_id, path=f"nodes[{in_id}].id", before=in_id, after=out_id,
                extra={"score": round(score, 3)},
            ))

    for in_id in in_ids:
        if in_id not in m.forward:
            node = inp.node_by_id[in_id]
            m.removed.append(in_id)
            m.matches.append(NodeMatch(in_id, None, method="unmatched"))
            m.findings.append(Finding(
                policy.severity("node_removed"), "node_removed",
                f"Node '{node.get('name')}' ({in_id}, {node.get('type')}) is missing from the output.",
                node_id=in_id, path=f"nodes[{in_id}]", before=node.get("name"),
            ))
    for out_id in out_ids:
        if out_id not in m.forward.values():
            node = out.node_by_id[out_id]
            m.added.append(out_id)
            m.matches.append(NodeMatch(None, out_id, method="unmatched"))
            m.findings.append(Finding(
                policy.severity("node_added"), "node_added",
                f"Output has a new node '{node.get('name')}' ({out_id}, {node.get('type')}) that is not in the input.",
                node_id=None, path=f"output.nodes[{out_id}]", after=node.get("name"),
            ))

    for match in m.matches:
        if match.input_id:
            match.input_name = inp.node_by_id[match.input_id].get("name")
        if match.output_id:
            match.output_name = out.node_by_id[match.output_id].get("name")
    order = {i: n for n, i in enumerate(in_ids)}
    m.matches.sort(key=lambda x: (x.input_id is None, order.get(x.input_id, 0), x.output_id or ""))
    return m
