"""report.json and report.md writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loader import condition_text, edge_destination, edges_of
from .model import Finding, Report, Severity, Verdict

TITLES = {
    "missing_key": "Required field missing",
    "invalid_value": "Invalid value",
    "empty_node_id": "Node without an id",
    "duplicate_node_id": "Two nodes share an id",
    "duplicate_edge_id": "Two transitions share an id",
    "start_node_missing": "Start node does not exist",
    "dangling_reference": "Transition points to a node that does not exist",
    "switch_case_invalid": "Switch case uses an unknown transition",
    "switch_default_invalid": "Switch default uses an unknown transition",
    "switch_expression_missing": "Switch node has no expression",
    "end_node_has_edges": "End node has outgoing transitions",
    "dead_end_node": "Node has no way out",
    "unreachable_node": "Node can no longer be reached",
    "no_path_to_end": "No path reaches an end node",
    "duplicate_global_variable": "Global variable declared twice",
    "node_id_shift": "Node ids were shifted",
    "node_id_changed": "Node id changed",
    "node_removed": "Node removed",
    "node_added": "Node added",
    "node_type_changed": "Node type changed",
    "node_name_changed": "Node renamed",
    "start_node_changed": "Start node changed",
    "edge_removed": "Transition removed",
    "edge_added": "Transition added",
    "edge_redirected": "Transition now leads somewhere else",
    "condition_changed": "Transition condition reworded",
    "condition_emptied": "Transition condition erased",
    "condition_whitespace_only": "Transition condition: spacing only",
    "edge_id_changed": "Transition id changed",
    "edge_order_changed": "Transitions reordered",
    "go_back_to_previous_changed": "Go-back behaviour changed",
    "topology_equivalent": "Graph still equivalent",
    "topology_not_equivalent": "Graph is no longer equivalent",
    "switch_expression_changed": "Switch expression changed",
    "switch_case_changed": "Switch routing changed",
    "switch_default_changed": "Switch default route changed",
    "global_variable_changed": "Global variables changed",
    "supported_modes_changed": "Supported modes changed",
    "allow_go_back_changed": "allowGoBack changed",
    "go_back_edges_changed": "Go-back transitions changed",
    "jump_to_edges_changed": "Jump transitions changed",
    "global_node_changed": "Global-node setting changed",
    "token_lost": "Placeholder, number, amount, time or link lost",
    "token_added": "New placeholder, number, amount, time or link",
    "overlay_language_removed": "Translation removed",
    "overlay_language_added": "Translation added",
    "prompt_key_removed": "Prompt removed",
    "prompt_key_added": "Prompt added",
    "text_whitespace_only": "Text: spacing only",
    "text_normalization_only": "Text: encoding form only",
    "text_truncated": "Text cut off",
    "text_corrupted": "Text corrupted",
    "text_needs_review": "Text rewritten (needs review)",
    "meaning_changed": "Meaning changed (AI judge)",
    "content_removed": "Content removed (AI judge)",
    "content_added": "Content added (AI judge)",
    "corrupted": "Text corrupted (AI judge)",
    "rewording_same_meaning": "Reworded, same meaning (AI judge)",
    "identical_meaning": "Same meaning (AI judge)",
    "needs_human_review": "Uncertain change (needs human review)",
    "position_changed": "Layout positions changed",
    "exported_at_changed": "Export timestamp changed",
    "node_reordered": "Nodes listed in a different order",
    "key_order_changed": "Field order changed",
    "nodes_container_changed": "Node list format changed",
    "envelope_changed": "File wrapper changed",
    "schema_key_added": "Unexpected field added",
    "schema_key_removed": "Field removed",
    "type_drift": "Value type changed",
    "field_changed": "Setting changed",
}

ICON = {Severity.CRITICAL: "❌", Severity.WARNING: "⚠️", Severity.INFO: "ℹ️"}
BANNER = {
    Verdict.PASS: ("✅", "PASS", "The output flow matches the input. Nothing needs attention."),
    Verdict.PASS_WITH_WARNINGS: ("⚠️", "PASS WITH WARNINGS",
                                 "Nothing is clearly broken, but some changes should be looked at by a person."),
    Verdict.FAIL: ("❌", "FAIL", "The output flow is damaged or behaves differently. Do not use it until the critical problems are fixed."),
}
SEVERITY_MEANING = {
    Severity.CRITICAL: "Breaks the flow or changes what the agent does. Must be fixed.",
    Severity.WARNING: "Might matter. A person should check it.",
    Severity.INFO: "Harmless (layout, spacing, formatting). No action needed.",
}


def write_json(report: Report, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _clip(value: Any, limit: int = 700) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > limit:
        text = text[:limit] + f"\n… ({len(text) - limit} more characters)"
    return text


def _fence(text: str, lang: str = "") -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}{lang}\n{text}\n{fence}"


def _cell(text: Any) -> str:
    return str(text if text is not None else "—").replace("|", "\\|").replace("\n", " ")


def _finding_md(n: int, f: Finding, names: dict[str, str]) -> list[str]:
    title = TITLES.get(f.category, f.category)
    node = f" · node **{_cell(names.get(f.node_id, f.node_id))}**" if f.node_id else ""
    judge = " · _AI judge_" if f.source == "llm" else ""
    lines = [f"#### {n}. {title}{node}{judge}", "", f.message, ""]
    if f.path:
        lines += [f"<sub>Where: `{f.path}` · category `{f.category}` · id `{f.id}`</sub>", ""]
    if f.extra.get("evidence"):
        lines += ["Evidence: " + " · ".join(f"“{e}”" for e in f.extra["evidence"]), ""]
    show_values = f.before is not None or f.after is not None
    if f.unified_diff:
        is_inline = f.category in ("condition_changed",)
        if is_inline:
            lines += ["Change (`[-removed-]` `{+added+}`):", "", _fence(_clip(f.unified_diff, 1500)), ""]
        else:
            lines += ["<details><summary>Show diff</summary>", "", _fence(_clip(f.unified_diff, 4000), "diff"), "", "</details>", ""]
    if show_values and (not f.unified_diff or f.category in ("condition_changed",)):
        if f.before is not None:
            lines += ["**Before:**", "", _fence(_clip(f.before)), ""]
        if f.after is not None:
            lines += ["**After:**", "", _fence(_clip(f.after)), ""]
    return lines


def _mermaid(report: Report) -> list[str]:
    graph = report.output_graph or {}
    flow = graph.get("flow")
    if flow is None or not flow.nodes or len(flow.nodes) > 80:
        return []
    changed = set(graph.get("changed_edges") or [])
    ids = {n.get("id"): f"n{i}" for i, n in enumerate(flow.nodes)}

    def label(text: str, limit: int = 40) -> str:
        text = " ".join(str(text).split())
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        for ch in '"[]{}()<>|#;`':
            text = text.replace(ch, " ")
        return text.strip() or "-"

    lines = ["```mermaid", "flowchart LR"]
    for node in flow.nodes:
        key, name = ids[node.get("id")], label(node.get("name") or node.get("id"), 30)
        shape = {"switch": ('{{"', '"}}'), "end": ('(["', '"])')}.get(node.get("type"), ('["', '"]'))
        lines.append(f"  {key}{shape[0]}{name}{shape[1]}")
    styles, index = [], 0
    for node in flow.nodes:
        for edge in edges_of(node):
            dest = edge_destination(node, edge)
            if dest not in ids:
                continue
            text = condition_text(edge)
            if not text and node.get("type") == "switch":
                cases = [k for k, v in (node.get("switchCases") or {}).items() if v == edge.get("id")]
                text = ", ".join(cases) or ("default" if node.get("switchDefaultEdgeId") == edge.get("id") else "")
            mark = "CHANGED: " if str(edge.get("id")) in changed else ""
            lines.append(f'  {ids[node.get("id")]} -->|"{mark}{label(text)}"| {ids[dest]}')
            if mark:
                styles.append(f"  linkStyle {index} stroke:#d62728,stroke-width:3px")
            index += 1
    lines += styles
    start = flow.start_node_id
    if start in ids:
        lines.append(f"  style {ids[start]} stroke-width:3px")
    lines.append("```")
    return lines


def render_markdown(report: Report) -> str:
    icon, word, meaning = BANNER[report.verdict]
    counts = report.counts
    names = {m.input_id: m.input_name or m.input_id for m in report.node_mapping if m.input_id}
    out = [
        "# Conversation flow integrity report",
        "",
        f"> ## {icon} {word}",
        f"> {meaning}",
        "",
        f"- **Original (input):** `{report.input_label}`",
        f"- **Copy to check (output):** `{report.output_label}`",
        f"- **Conversation graph equivalent:** {'yes' if report.topology_equivalent else 'no'}",
        f"- **AI judge for rewritten text:** {report.judge_status}" + (f" (`{report.judge_model}`)" if report.judge_model else ""),
        "",
        "## Summary",
        "",
        "| Level | Count | What it means |",
        "| --- | ---: | --- |",
    ]
    for sev in (Severity.CRITICAL, Severity.WARNING, Severity.INFO):
        out.append(f"| {ICON[sev]} {sev.value.title()} | {counts[sev.value]} | {SEVERITY_MEANING[sev]} |")
    out.append("")

    if report.findings:
        by_cat: dict[str, list[Finding]] = {}
        for f in report.findings:
            by_cat.setdefault(f.category, []).append(f)
        out += ["**At a glance**", ""]
        for cat, items in by_cat.items():
            out.append(f"- {ICON[items[0].severity]} {TITLES.get(cat, cat)}: {len(items)}")
        out.append("")

    if report.judge_summary:
        out += ["## In plain English", "", "_Written by the AI judge from the findings below._", ""]
        out += [f"- {b}" for b in report.judge_summary]
        out.append("")
    elif report.judge_status.startswith("skipped"):
        out += ["_The AI judge was not run, so rewritten text is listed as “needs review” instead of being classified._", ""]

    out += ["## Node mapping", "",
            "How each node in the original was matched to a node in the copy.", "",
            "| Original node | Original id | Copy id | Matched by | Similarity |",
            "| --- | --- | --- | --- | ---: |"]
    method_text = {"id": "same id", "content": "content (id changed)", "id_shift": "content (ids shifted!)",
                   "unmatched": "not matched"}
    for m in report.node_mapping:
        name = m.input_name if m.input_id else f"(new) {m.output_name}"
        score = f"{m.score:.2f}" if m.score is not None else "—"
        out.append(f"| {_cell(name)} | {_cell(m.input_id)} | {_cell(m.output_id)} | {method_text.get(m.method, m.method)} | {score} |")
    out.append("")

    out += ["## Findings", ""]
    if not report.findings:
        out += ["No differences found.", ""]
    for sev in (Severity.CRITICAL, Severity.WARNING, Severity.INFO):
        items = [f for f in report.findings if f.severity is sev]
        if not items:
            continue
        out += [f"### {ICON[sev]} {sev.value.title()} ({len(items)})", ""]
        for n, f in enumerate(items, 1):
            out += _finding_md(n, f, names)

    if report.input_issues:
        out += ["## Problems already present in the original", "",
                "These exist in the input too, so the tool did not cause them. They do not affect the verdict.", ""]
        for f in report.input_issues:
            out.append(f"- {ICON[f.severity]} **{TITLES.get(f.category, f.category)}**: {f.message}")
        out.append("")

    graph = _mermaid(report)
    if graph:
        out += ["## Output flow graph", "", "Transitions with a problem are drawn in red and marked CHANGED.", ""]
        out += graph
        out.append("")
    return "\n".join(out)


def write_markdown(report: Report, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def write_reports(report: Report, report_dir: str | Path) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    return write_json(report, report_dir / "report.json"), write_markdown(report, report_dir / "report.md")
