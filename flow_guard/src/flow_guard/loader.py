"""Load a flow from a path, JSON string or dict, and detect its envelope.

Accepted shapes:
  * full export  {version, type, exportedAt, data: {conversationNodes, ...}}
  * bare data    {conversationNodes, prompt, ...}
  * bare graph   {startNodeId, nodes, ...}   (the conversationNodes object itself)

`nodes` may be a list or a dict keyed by node id. The caller's object is never
mutated: everything is deep-copied first.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FlowLoadError(Exception):
    """The flow could not be read or is not a conversation-flow document."""


EDGE_GROUPS = ("edges", "goBackEdges", "jumpToEdges")


@dataclass
class Flow:
    raw: dict                 # the full document as loaded (deep copy)
    envelope: str             # "envelope" | "data" | "conversation_nodes"
    data: dict                # the `data` object (synthesised for bare graphs)
    conv: dict                # the conversationNodes object
    nodes: list[dict]         # nodes in document order
    nodes_container: str      # "list" | "dict"
    label: str = "flow"
    node_by_id: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for node in self.nodes:
            node_id = node.get("id")
            if isinstance(node_id, str) and node_id not in self.node_by_id:
                self.node_by_id[node_id] = node

    @property
    def start_node_id(self) -> Any:
        return self.conv.get("startNodeId")

    @property
    def global_variables(self) -> list:
        value = self.conv.get("globalVariables")
        return value if isinstance(value, list) else []


def edges_of(node: dict, group: str = "edges") -> list[dict]:
    value = node.get(group)
    if not isinstance(value, list):
        return []
    return [e for e in value if isinstance(e, dict)]


def edge_destination(node: dict, edge: dict, group: str = "edges") -> Any:
    """Where an edge leads. A jumpToEdge without destinationNodeId targets its
    own (global) node: it is the way *into* that node from anywhere."""
    dest = edge.get("destinationNodeId")
    if dest is None and group == "jumpToEdges":
        return node.get("id")
    return dest


def condition_text(edge: dict) -> str:
    cond = edge.get("transitionCondition")
    if isinstance(cond, dict):
        prompt = cond.get("prompt")
        return prompt if isinstance(prompt, str) else ""
    prompt = edge.get("prompt")
    return prompt if isinstance(prompt, str) else ""


def _read_source(source: Any) -> tuple[Any, str]:
    if isinstance(source, Flow):
        return copy.deepcopy(source.raw), source.label
    if isinstance(source, dict):
        return copy.deepcopy(source), "<dict>"
    if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith(("{", "["))):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise FlowLoadError(f"cannot read {path}: {exc.strerror or exc}") from exc
        try:
            return json.loads(text), str(path)
        except json.JSONDecodeError as exc:
            raise FlowLoadError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(source, (str, bytes)):
        try:
            return json.loads(source), "<string>"
        except json.JSONDecodeError as exc:
            raise FlowLoadError(f"input is not valid JSON: {exc}") from exc
    raise FlowLoadError(f"unsupported flow source type: {type(source).__name__}")


def load_flow(source: Any, label: str | None = None) -> Flow:
    doc, default_label = _read_source(source)
    label = label or default_label
    if not isinstance(doc, dict):
        raise FlowLoadError(f"{label}: top level must be a JSON object")

    if isinstance(doc.get("data"), dict) and "conversationNodes" in doc["data"]:
        envelope, data = "envelope", doc["data"]
    elif "conversationNodes" in doc:
        envelope, data = "data", doc
    elif "nodes" in doc and ("startNodeId" in doc or isinstance(doc.get("nodes"), (list, dict))):
        envelope, data = "conversation_nodes", {"conversationNodes": doc}
    else:
        raise FlowLoadError(f"{label}: no data.conversationNodes found; this is not a conversation-flow export")

    conv = data.get("conversationNodes")
    if not isinstance(conv, dict):
        raise FlowLoadError(f"{label}: conversationNodes must be an object")

    raw_nodes = conv.get("nodes")
    if isinstance(raw_nodes, list):
        container = "list"
        nodes = [n for n in raw_nodes if isinstance(n, dict)]
    elif isinstance(raw_nodes, dict):
        container = "dict"
        nodes = []
        for key, node in raw_nodes.items():
            if isinstance(node, dict):
                node.setdefault("id", key)  # safe: `doc` is our own deep copy
                nodes.append(node)
    else:
        raise FlowLoadError(f"{label}: conversationNodes.nodes must be a list or an object keyed by node id")

    return Flow(raw=doc, envelope=envelope, data=data, conv=conv, nodes=nodes,
                nodes_container=container, label=label)
