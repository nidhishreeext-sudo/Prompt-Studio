"""Integrity check of Prompt Studio's output flow against its input flow.

Runs flow_guard's deterministic rules only (no AI judge: flow_guard's judge is
OpenAI-based and Prompt Studio runs on Gemini), so it needs no OPENAI_API_KEY
and makes no network calls.

Severity policy. flow_guard reports every lost token (placeholder, URL,
number, time, currency) under the single category "token_lost", so its Policy
cannot give numbers a different severity from placeholders. Prompt Studio's
distill step deliberately deletes pronunciation rules, and those quote
numbers and times ("say 10:30 AM as ..."), so a correct distill would often
FAIL. Weakening "token_lost" globally would also hide a lost {{placeholder}},
so flow_guard's own report is left untouched and this module re-labels only
token_lost findings that (a) sit on a field Prompt Studio rewrote and (b) lost
nothing but numbers, times or currency amounts, from CRITICAL to WARNING.
Both verdicts are returned, and every re-labelled finding is listed with the
text around each removed figure so a person can confirm it was a language rule.
"""

import traceback

# Token kinds whose removal from a rewritten field is a WARNING, not CRITICAL.
# Placeholders and URLs are deliberately absent: losing one is always CRITICAL.
DOWNGRADABLE_TOKEN_KINDS = frozenset({"number", "time", "currency"})

_SEVERITY_RANK = {"CRITICAL": 2, "WARNING": 1, "INFO": 0}


def _load_flow_guard():
    import flow_guard
    # The repo folder flow_guard/ has no __init__.py, so without the installed
    # package `import flow_guard` silently resolves to it as a namespace package.
    if getattr(flow_guard, "__file__", None) is None:
        raise ImportError("flow_guard resolved to the bare repo folder, not the installed package "
                          "(run pip install -r requirements.txt)")
    from flow_guard import Policy, compare
    from flow_guard.report import TITLES
    return compare, Policy, TITLES


def _verdict(findings: list) -> str:
    severities = {f["severity"] for f in findings}
    if "CRITICAL" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _counts(findings: list) -> dict:
    counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
    for f in findings:
        counts[f["severity"]] += 1
    return counts


def _compact(f: dict, titles: dict) -> dict:
    item = {
        "id": f["id"],
        "severity": f["severity"],
        "category": f["category"],
        "title": titles.get(f["category"], f["category"]),
        "message": f["message"],
        "location": {"node_id": f.get("node_id"), "path": f.get("path") or ""},
    }
    tokens = (f.get("extra") or {}).get("tokens")
    if tokens:
        item["tokens"] = tokens
    return item


def _expected_reason(f: dict, rewritten: set, attached: set, created_prompts: bool) -> str | None:
    """Findings that are the intended effect of Prompt Studio's own changes.
    They keep their severity; the flag only tells the user they were expected."""
    path, category = f.get("path") or "", f["category"]
    if category == "prompt_key_added" and path.startswith("data.prompts.") and path[len("data.prompts."):] in attached:
        return "language prompt attached by Prompt Studio"
    if category == "text_needs_review" and path in rewritten:
        return "field rewritten by Prompt Studio; the AI judge is off, so every rewrite is listed for review"
    if created_prompts and path == "data.prompts" and category in ("schema_key_added", "field_changed"):
        return "data.prompts created by Prompt Studio to hold the language prompts"
    return None


def adjust_report(raw: dict, rewritten_paths, attached_keys, created_prompts: bool, titles: dict) -> dict:
    """Prompt Studio's view of a flow_guard report (see the module docstring)."""
    rewritten, attached = set(rewritten_paths), set(attached_keys)
    findings, downgraded = [], []
    for f in raw["findings"]:
        item = _compact(f, titles)
        tokens = item.get("tokens") or []
        if (f["category"] == "token_lost" and f["severity"] == "CRITICAL" and f.get("path") in rewritten
                and tokens and all(t.get("kind") in DOWNGRADABLE_TOKEN_KINDS for t in tokens)):
            item["severity"] = "WARNING"
            item["downgraded_from"] = "CRITICAL"
            downgraded.append({
                "id": f["id"], "path": f.get("path"), "node_id": f.get("node_id"),
                "from": "CRITICAL", "to": "WARNING",
                "reason": ("Only numbers, times or amounts were removed, from a field Prompt Studio distilled. "
                           "Confirm from the context that each was part of a language rule, not a business fact."),
                "message": f["message"],
                "tokens": tokens,
            })
        reason = _expected_reason(f, rewritten, attached, created_prompts)
        if reason:
            item["expected"] = reason
        findings.append(item)
    findings.sort(key=lambda x: -_SEVERITY_RANK[x["severity"]])  # stable: keeps flow_guard's order within a level

    return {
        "verdict": _verdict(findings),
        "raw_verdict": raw["verdict"],
        "counts": _counts(findings),
        "raw_counts": raw["counts"],
        "topology_equivalent": raw["topology_equivalent"],
        "findings": findings,
        "preexisting_problems": [_compact(f, titles) for f in raw.get("input_issues") or []],
        "downgraded": downgraded,
        "judge": raw.get("judge", {}).get("status"),
        "raw_report": raw,  # flow_guard's own report.json content, for download
    }


def run_flow_check(input_flow: dict, output_flow: dict, rewritten_paths=(), attached_keys=(),
                   created_prompts: bool = False) -> dict:
    """Never raises: any failure becomes {"error": ...} so a guard problem can
    never turn a successful generation into a failed request."""
    try:
        compare, Policy, titles = _load_flow_guard()
        report = compare(input_flow, output_flow, policy=Policy(), judge=None, summarize=False)
        # Report.to_dict() is exactly what flow_guard's write_reports puts in report.json.
        return adjust_report(report.to_dict(), rewritten_paths, attached_keys, created_prompts, titles)
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
