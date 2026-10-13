"""Command line: `flow-guard check INPUT OUTPUT` and `flow-guard validate FLOW`."""

from __future__ import annotations

import argparse
import sys

from . import compare, validate
from .judge import judge_from_env
from .loader import FlowLoadError, load_flow
from .model import Severity, Verdict
from .policy import Policy, PolicyError, load_policy
from .report import ICON, write_reports

EXIT_PASS, EXIT_WARN, EXIT_FAIL, EXIT_USAGE = 0, 1, 2, 3


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # usage errors exit 3, not argparse's 2
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="flow-guard", description="Integrity checker and diff tool for conversation-flow JSON.")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    check = sub.add_parser("check", help="compare an input flow with an output flow")
    check.add_argument("input", help="the original flow JSON")
    check.add_argument("output", help="the flow JSON produced by the tool")
    check.add_argument("--policy", help="JSON file overriding severities/thresholds")
    check.add_argument("--no-llm", action="store_true", help="do not call the OpenAI judge")
    check.add_argument("--model", help="judge model (default: $OPENAI_JUDGE_MODEL or gpt-6-luna)")
    check.add_argument("--report-dir", default="flow_guard_report", help="where to write report.json and report.md")
    check.add_argument("--strict-ids", action="store_true", help="treat any node/edge id change as CRITICAL")
    check.add_argument("--strict-schema", action="store_true", help="treat added/removed keys as CRITICAL")
    check.add_argument("--fail-on-warning", action="store_true", help="exit 2 instead of 1 on PASS_WITH_WARNINGS")

    val = sub.add_parser("validate", help="structurally validate one flow")
    val.add_argument("flow")
    val.add_argument("--policy", help="JSON file overriding severities")
    return parser


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _policy(args) -> Policy:
    policy = load_policy(args.policy) if args.policy else Policy()
    return policy.with_flags(strict_ids=getattr(args, "strict_ids", False) or None,
                             strict_schema=getattr(args, "strict_schema", False) or None)


def run_check(args) -> int:
    policy = _policy(args)
    inp = load_flow(args.input, label=args.input)
    out = load_flow(args.output, label=args.output)
    if args.no_llm:
        judge, status = None, "skipped: --no-llm"
    else:
        judge, status = judge_from_env(args.model)
    report = compare(inp, out, policy=policy, judge=judge)
    if judge is None:
        report.judge_status = status
    json_path, md_path = write_reports(report, args.report_dir)

    verdict = report.verdict
    c = report.counts
    print(f"{verdict.value}  (critical {c['CRITICAL']}, warning {c['WARNING']}, info {c['INFO']})")
    for f in report.findings:
        if f.severity is not Severity.INFO:
            print(f"  {ICON[f.severity]} [{f.category}] {f.message[:200]}")
    print(f"AI judge: {report.judge_status}")
    print(f"Reports: {json_path}  {md_path}")
    if verdict is Verdict.FAIL:
        return EXIT_FAIL
    if verdict is Verdict.PASS_WITH_WARNINGS:
        return EXIT_FAIL if args.fail_on_warning else EXIT_WARN
    return EXIT_PASS


def run_validate(args) -> int:
    policy = _policy(args)
    findings = validate(load_flow(args.flow, label=args.flow), policy)
    if not findings:
        print("VALID: no structural problems found.")
        return EXIT_PASS
    for f in findings:
        print(f"{ICON[f.severity]} {f.severity.value} [{f.category}] {f.message}")
    return EXIT_FAIL if any(f.severity is Severity.CRITICAL for f in findings) else EXIT_WARN


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return run_check(args) if args.command == "check" else run_validate(args)
    except (FlowLoadError, PolicyError) as exc:
        print(f"flow-guard: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"flow-guard: I/O error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
