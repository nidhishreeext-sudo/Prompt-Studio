"""A. Single-flow structural validation."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from .loader import EDGE_GROUPS, Flow, edge_destination, edges_of, load_flow
from .model import Finding
from .policy import Policy

NODE_TYPES = ("conversation", "switch", "end")


def _f(policy: Policy, category: str, message: str, node_id: Any = None, path: str = "", key: str = "", **kw) -> Finding:
    extra = {"key": key} if key else {}
    return Finding(policy.severity(category), category, message,
                   node_id=node_id if isinstance(node_id, str) else None, path=path, extra=extra, **kw)


def node_label(node: dict) -> str:
    name = node.get("name")
    return f"'{name}' ({node.get('id')})" if name else f"'{node.get('id')}'"


def outgoing(flow: Flow, node: dict) -> list[str]:
    """Destination ids of every outgoing transition of a node."""
    dests = []
    for group in ("edges", "goBackEdges"):
        for edge in edges_of(node, group):
            dest = edge_destination(node, edge, group)
            if isinstance(dest, str):
                dests.append(dest)
    for edge in edges_of(node, "jumpToEdges"):
        dest = edge.get("destinationNodeId")
        if isinstance(dest, str):
            dests.append(dest)
    return dests


def reachable_from_start(flow: Flow) -> set[str]:
    start = flow.start_node_id
    if start not in flow.node_by_id:
        return set()
    # Global nodes (globalNode.enabled) can be jumped to from anywhere.
    global_ids = [n["id"] for n in flow.nodes
                  if isinstance(n.get("id"), str) and (n.get("globalNode") or {}).get("enabled") is True]
    seen = {start}
    queue = deque([start])
    globals_added = False
    while queue:
        node_id = queue.popleft()
        for dest in outgoing(flow, flow.node_by_id[node_id]):
            if dest in flow.node_by_id and dest not in seen:
                seen.add(dest)
                queue.append(dest)
        if not queue and not globals_added:
            globals_added = True
            for gid in global_ids:
                if gid not in seen:
                    seen.add(gid)
                    queue.append(gid)
    return seen


def validate(flow: Any, policy: Policy | None = None) -> list[Finding]:
    """Validate one flow on its own. Accepts a Flow, dict, JSON string or path."""
    policy = policy or Policy()
    if not isinstance(flow, Flow):
        flow = load_flow(flow)
    findings: list[Finding] = []
    add = findings.append

    if flow.envelope == "envelope":
        for key in ("version", "type"):
            if key not in flow.raw:
                add(_f(policy, "missing_key", f"Top-level key '{key}' is missing.", path=key, key=key))
        if flow.raw.get("type") not in (None, "guided_conversation_flow"):
            add(_f(policy, "invalid_value", f"Top-level type is {flow.raw.get('type')!r}, expected 'guided_conversation_flow'.",
                   path="type", key="type"))
    for key in ("startNodeId", "nodes"):
        if key not in flow.conv:
            add(_f(policy, "missing_key", f"conversationNodes.{key} is missing.", path=f"conversationNodes.{key}", key=key))

    # --- node ids ---------------------------------------------------------------
    id_counts = Counter(n.get("id") for n in flow.nodes)
    for index, node in enumerate(flow.nodes):
        node_id = node.get("id")
        path = f"nodes[{node_id if isinstance(node_id, str) and node_id else index}]"
        if not isinstance(node_id, str) or not node_id.strip():
            add(_f(policy, "empty_node_id", f"Node #{index + 1} ({node.get('name')!r}) has no id.", path=path, key=str(index)))
        for key in ("id", "type", "edges"):
            if key not in node and not (key == "edges" and node.get("type") == "end"):
                add(_f(policy, "missing_key", f"Node {node_label(node)} is missing '{key}'.", node_id, f"{path}.{key}", key=key))
        if node.get("type") not in NODE_TYPES and "type" in node:
            add(_f(policy, "invalid_value", f"Node {node_label(node)} has unknown type {node.get('type')!r}.",
                   node_id, f"{path}.type", key="type"))
    for node_id, count in sorted(((k, v) for k, v in id_counts.items() if isinstance(k, str) and k and v > 1)):
        add(_f(policy, "duplicate_node_id", f"Node id '{node_id}' is used by {count} nodes.",
               node_id, f"nodes[{node_id}]", key=node_id))

    # --- edge ids (unique across the whole flow) ---------------------------------
    edge_owners: dict[str, list[str]] = {}
    for node in flow.nodes:
        for group in EDGE_GROUPS:
            for edge in edges_of(node, group):
                if isinstance(edge.get("id"), str):
                    edge_owners.setdefault(edge["id"], []).append(str(node.get("id")))
                elif group == "edges":
                    add(_f(policy, "missing_key", f"An edge of node {node_label(node)} has no id.",
                           node.get("id"), f"nodes[{node.get('id')}].edges", key="edge.id"))
    for edge_id, owners in sorted(edge_owners.items()):
        if len(owners) > 1:
            add(_f(policy, "duplicate_edge_id", f"Edge id '{edge_id}' is used {len(owners)} times (nodes: {', '.join(owners)}).",
                   owners[0], f"edges[{edge_id}]", key=edge_id))

    # --- start node -------------------------------------------------------------
    start = flow.start_node_id
    if "startNodeId" in flow.conv and start not in flow.node_by_id:
        add(_f(policy, "start_node_missing", f"startNodeId '{start}' does not match any node.",
               path="conversationNodes.startNodeId", key="start"))

    # --- per-node edge checks ----------------------------------------------------
    for node in flow.nodes:
        node_id = node.get("id")
        base = f"nodes[{node_id}]"
        node_type = node.get("type")
        for group in EDGE_GROUPS:
            for edge in edges_of(node, group):
                dest = edge.get("destinationNodeId")
                if dest is None and group == "jumpToEdges":
                    continue
                if dest not in flow.node_by_id:
                    add(_f(policy, "dangling_reference",
                           f"{group[:-1] if group != 'edges' else 'Edge'} '{edge.get('id')}' of node {node_label(node)} "
                           f"points to node '{dest}', which does not exist.",
                           node_id, f"{base}.{group}[{edge.get('id')}].destinationNodeId",
                           key=f"{group}:{edge.get('id')}", after=dest))

        edge_ids = {e.get("id") for e in edges_of(node)}
        if node_type == "switch":
            if not str(node.get("switchExpression") or "").strip():
                add(_f(policy, "switch_expression_missing", f"Switch node {node_label(node)} has no switchExpression.",
                       node_id, f"{base}.switchExpression", key="switchExpression"))
            cases = node.get("switchCases") or {}
            if isinstance(cases, dict):
                for case, edge_id in sorted(cases.items()):
                    if edge_id not in edge_ids:
                        add(_f(policy, "switch_case_invalid",
                               f"Switch case '{case}' of node {node_label(node)} uses edge '{edge_id}', which is not one of this node's edges.",
                               node_id, f"{base}.switchCases.{case}", key=f"case:{case}", after=edge_id))
            default = node.get("switchDefaultEdgeId")
            if default is not None and default not in edge_ids:
                add(_f(policy, "switch_default_invalid",
                       f"switchDefaultEdgeId '{default}' of node {node_label(node)} is not one of this node's edges.",
                       node_id, f"{base}.switchDefaultEdgeId", key="default", after=default))
        if node_type == "end":
            if edges_of(node):
                add(_f(policy, "end_node_has_edges", f"End node {node_label(node)} has {len(edges_of(node))} outgoing edge(s).",
                       node_id, f"{base}.edges", key="end_edges"))
        elif node_type in ("conversation", "switch") and not edges_of(node) and not edges_of(node, "goBackEdges"):
            add(_f(policy, "dead_end_node", f"Node {node_label(node)} is not an end node but has no outgoing edges.",
                   node_id, f"{base}.edges", key="dead_end"))

    # --- reachability ------------------------------------------------------------
    if start in flow.node_by_id:
        reachable = reachable_from_start(flow)
        for node in flow.nodes:
            if isinstance(node.get("id"), str) and node["id"] not in reachable:
                add(_f(policy, "unreachable_node", f"Node {node_label(node)} cannot be reached from the start node.",
                       node["id"], f"nodes[{node['id']}]", key="unreachable"))
        if not any(flow.node_by_id[n].get("type") == "end" for n in reachable):
            add(_f(policy, "no_path_to_end", "No path from the start node reaches an end node.",
                   path="conversationNodes", key="no_end"))

    # --- global variables ----------------------------------------------------------
    names = Counter(v.get("name") for v in flow.global_variables if isinstance(v, dict))
    for name, count in sorted(((k, v) for k, v in names.items() if isinstance(k, str) and v > 1)):
        add(_f(policy, "duplicate_global_variable", f"Global variable '{name}' is declared {count} times.",
               path=f"globalVariables[{name}]", key=name))

    return findings
