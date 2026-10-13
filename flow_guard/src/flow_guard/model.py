"""Core result types: Severity, Finding, Report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}[self.value]

    @classmethod
    def parse(cls, value: "str | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        try:
            return cls(str(value).upper())
        except ValueError:
            raise ValueError(f"unknown severity {value!r}; use INFO, WARNING or CRITICAL") from None


class Verdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


@dataclass
class Finding:
    severity: Severity
    category: str
    message: str
    node_id: str | None = None
    path: str = ""
    before: Any = None
    after: Any = None
    source: str = "deterministic"
    unified_diff: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        basis = json.dumps(
            [self.category, self.node_id, self.path, self.message, self.source],
            ensure_ascii=False,
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]

    def sort_key(self) -> tuple:
        return (-self.severity.rank, self.category, self.node_id or "", self.path, self.message)

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category,
            "node_id": self.node_id,
            "path": self.path,
            "message": self.message,
            "before": self.before,
            "after": self.after,
            "source": self.source,
            "unified_diff": self.unified_diff,
        }
        if self.extra:
            out["extra"] = self.extra
        return out


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Stable, deterministic order: most severe first, then category/node/path."""
    return sorted(findings, key=Finding.sort_key)


def verdict_for(findings: list[Finding]) -> Verdict:
    severities = {f.severity for f in findings}
    if Severity.CRITICAL in severities:
        return Verdict.FAIL
    if Severity.WARNING in severities:
        return Verdict.PASS_WITH_WARNINGS
    return Verdict.PASS


@dataclass
class NodeMatch:
    input_id: str | None
    output_id: str | None
    input_name: str | None = None
    output_name: str | None = None
    method: str = "id"  # id | content | id_shift | unmatched
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "input_id": self.input_id,
            "output_id": self.output_id,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "method": self.method,
            "score": None if self.score is None else round(self.score, 3),
        }


@dataclass
class Report:
    findings: list[Finding]
    node_mapping: list[NodeMatch] = field(default_factory=list)
    topology_equivalent: bool | None = None
    input_issues: list[Finding] = field(default_factory=list)
    judge_status: str = "not run"
    judge_model: str | None = None
    judge_summary: list[str] = field(default_factory=list)
    input_label: str = "input"
    output_label: str = "output"
    output_graph: dict | None = None  # data for the optional mermaid diagram

    def __post_init__(self) -> None:
        self.findings = sort_findings(self.findings)
        self.input_issues = sort_findings(self.input_issues)

    @property
    def verdict(self) -> Verdict:
        return verdict_for(self.findings)

    @property
    def counts(self) -> dict:
        counts = {s.value: 0 for s in (Severity.CRITICAL, Severity.WARNING, Severity.INFO)}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    def by_category(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    def categories(self) -> set[str]:
        return {f.category for f in self.findings}

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict.value,
            "counts": self.counts,
            "input": self.input_label,
            "output": self.output_label,
            "topology_equivalent": self.topology_equivalent,
            "judge": {
                "status": self.judge_status,
                "model": self.judge_model,
                "summary": self.judge_summary,
            },
            "node_mapping": [m.to_dict() for m in self.node_mapping],
            "findings": [f.to_dict() for f in self.findings],
            "input_issues": [f.to_dict() for f in self.input_issues],
        }
