"""flow_guard: integrity checker and diff tool for guided-conversation-flow JSON.

    from flow_guard import compare, validate, assert_flow_preserved

    report = compare("before.json", "after.json")      # deterministic only
    report.verdict                                      # Verdict.PASS / PASS_WITH_WARNINGS / FAIL
"""

from __future__ import annotations

from typing import Any

from .diff import PairChecker, TextChange
from .judge import CLASSIFICATIONS, Judge, JudgeError, OpenAIJudge, judge_from_env
from .loader import Flow, FlowLoadError, load_flow
from .matcher import Mapping, match_nodes
from .model import Finding, NodeMatch, Report, Severity, Verdict, sort_findings
from .policy import Policy, PolicyError, load_policy
from .structure import validate as _validate

__all__ = [
    "compare", "validate", "assert_flow_preserved", "FlowIntegrityError",
    "Finding", "Severity", "Verdict", "Report", "Policy", "load_policy", "PolicyError",
    "FlowLoadError", "OpenAIJudge", "Judge", "JudgeError", "judge_from_env", "load_flow",
]
__version__ = "0.1.0"


class FlowIntegrityError(Exception):
    """Raised by assert_flow_preserved when the output flow FAILs the check."""

    def __init__(self, report: Report):
        self.report = report
        critical = [f for f in report.findings if f.severity is Severity.CRITICAL]
        lines = "\n".join(f"  - [{f.category}] {f.message}" for f in critical[:10])
        more = f"\n  ... and {len(critical) - 10} more" if len(critical) > 10 else ""
        super().__init__(f"conversation flow integrity check FAILED ({len(critical)} critical):\n{lines}{more}")


def validate(flow: Any, policy: Policy | None = None) -> list[Finding]:
    """Structural validation of a single flow (dict, JSON string, path or Flow)."""
    return sort_findings(_validate(flow, policy))


def _preexisting_key(f: Finding, translate) -> tuple:
    return (f.category, translate(f.node_id), f.extra.get("key", f.path))


def _structure_findings(inp: Flow, out: Flow, mapping: Mapping, policy: Policy) -> tuple[list[Finding], list[Finding]]:
    """Output structure problems that the input did not already have, plus the
    input's own problems (reported separately, never counted in the verdict)."""
    in_issues = _validate(inp, policy)
    out_issues = _validate(out, policy)
    known = {_preexisting_key(f, mapping.map_id if f.node_id else (lambda x: x)) for f in in_issues}
    new = [f for f in out_issues if _preexisting_key(f, lambda x: x) not in known]
    return new, in_issues


def _judge_texts(changes: list[TextChange], findings: list[Finding], judge: Judge | None,
                 policy: Policy) -> tuple[list[Finding], str | None]:
    """Turn text changes into findings, using the judge when available."""
    out: list[Finding] = []
    error = None
    groups: dict[str, list[TextChange]] = {}
    for change in changes:
        groups.setdefault(change.group, []).append(change)
    min_conf = policy.threshold("judge_min_confidence")

    def needs_review(change: TextChange, why: str) -> Finding:
        where = change.where or change.path
        return Finding(policy.severity("text_needs_review"), "text_needs_review",
                       f"{where} was rewritten; {why}. A person should read the diff.",
                       node_id=change.node_id, path=change.path, before=change.before, after=change.after,
                       unified_diff=change.unified_diff)

    for group in sorted(groups):
        batch = groups[group]
        if judge is None:
            out.extend(needs_review(c, "the AI judge was not run") for c in batch)
            continue
        fields = [{"field_id": f"f{i + 1}", "path": c.path, "before": c.before, "after": c.after, "diff": c.unified_diff}
                  for i, c in enumerate(batch)]
        node = {"id": batch[0].node_id, "name": batch[0].node_name} if batch[0].node_id else {"id": None, "name": "global settings"}
        context = [f"{f.severity.value} {f.category}: {f.message}" for f in findings
                   if (f.node_id == batch[0].node_id) and f.source == "deterministic"]
        try:
            verdicts = {str(v.get("field_id")): v for v in judge.review_fields(node, fields, context) if isinstance(v, dict)}
        except Exception as exc:  # the judge is optional; never let it crash the check
            error = str(exc)
            out.extend(needs_review(c, f"the AI judge failed ({str(exc)[:120]})") for c in batch)
            continue
        for field, change in zip(fields, batch):
            v = verdicts.get(field["field_id"])
            cls = v.get("classification") if v else None
            if cls not in CLASSIFICATIONS:
                out.append(needs_review(change, "the AI judge gave no usable answer"))
                continue
            try:
                conf = max(0.0, min(1.0, float(v.get("confidence", 0))))
            except (TypeError, ValueError):
                conf = 0.0
            evidence = [str(e) for e in (v.get("evidence") or [])][:6]
            rules = [str(r) for r in (v.get("affected_rules") or [])][:10]
            category = cls
            severity = policy.severity(cls)
            where = change.where or change.path
            label = {
                "identical_meaning": "same meaning, different formatting",
                "rewording_same_meaning": "reworded; the AI judge found the meaning unchanged",
                "meaning_changed": "the meaning changed",
                "content_removed": "content was removed",
                "content_added": "content was added",
                "corrupted": "the text looks corrupted",
            }[cls]
            message = f"{where}: {label} (judge confidence {conf:.2f})."
            if rules:
                message += " Could affect: " + "; ".join(rules) + "."
            if severity is Severity.CRITICAL and conf < min_conf:
                category, severity = "needs_human_review", policy.severity("needs_human_review")
                message = f"{where}: the AI judge suspects {label} but is unsure (confidence {conf:.2f}); needs human review."
            out.append(Finding(severity, category, message, node_id=change.node_id, path=change.path,
                               before=change.before, after=change.after, source="llm",
                               unified_diff=change.unified_diff,
                               extra={"classification": cls, "confidence": conf, "evidence": evidence,
                                      "affected_rules": rules,
                                      **({"detail_level": "INFO"} if cls == "rewording_same_meaning" else {})}))
    return out, error


def _digest(report: Report) -> dict:
    return {
        "verdict": report.verdict.value,
        "counts": report.counts,
        "topology_equivalent": report.topology_equivalent,
        "findings": [{"severity": f.severity.value, "category": f.category, "node": f.node_id,
                      "path": f.path, "message": f.message[:400]} for f in report.findings[:80]],
    }


def compare(input_flow: Any, output_flow: Any, policy: Policy | None = None,
            judge: Judge | None = None, summarize: bool = True) -> Report:
    """Compare an input flow with an output flow.

    Flows may be dicts, JSON strings, paths or Flow objects; none is modified.
    `judge=None` runs deterministic checks only (changed text becomes a
    "needs review" WARNING). Pass `OpenAIJudge()` (or any object with
    `review_fields` and `summarize`) to classify text changes.
    """
    policy = policy or Policy()
    inp = input_flow if isinstance(input_flow, Flow) else load_flow(input_flow, label="input")
    out = output_flow if isinstance(output_flow, Flow) else load_flow(output_flow, label="output")

    mapping = match_nodes(inp, out, policy)
    structure_new, input_issues = _structure_findings(inp, out, mapping, policy)
    checker = PairChecker(inp, out, mapping, policy).run()

    findings = mapping.findings + structure_new + checker.findings
    text_findings, judge_error = _judge_texts(checker.text_changes, findings, judge, policy)
    findings += text_findings

    if judge is None:
        status = "skipped"
    elif judge_error:
        status = f"partial: {judge_error[:200]}"
    else:
        status = "enabled"
    report = Report(
        findings=findings, node_mapping=mapping.matches, topology_equivalent=checker.topology_equivalent,
        input_issues=input_issues, judge_status=status, judge_model=getattr(judge, "model", None),
        input_label=inp.label, output_label=out.label,
        output_graph={"flow": out, "changed_edges": sorted(checker.changed_output_edges)},
    )
    if judge is not None and summarize and report.findings:
        try:
            report.judge_summary = judge.summarize(_digest(report))[:10]
        except Exception as exc:
            report.judge_status = f"{report.judge_status}; summary failed: {str(exc)[:150]}"
    return report


def assert_flow_preserved(before: Any, after: Any, policy: Policy | None = None, judge: Judge | None = None) -> Report:
    """Pipeline guard: returns the Report, raises FlowIntegrityError on FAIL."""
    report = compare(before, after, policy=policy, judge=judge)
    if report.verdict is Verdict.FAIL:
        raise FlowIntegrityError(report)
    return report
