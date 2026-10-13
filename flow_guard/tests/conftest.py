import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "tests" / "fixtures" / "synthetic_flow.json"
SAMPLE = ROOT / "samples" / "input_flow.json"

FLOW_FILES = [SYNTHETIC] + ([SAMPLE] if SAMPLE.exists() else [])


@pytest.fixture(params=FLOW_FILES, ids=[p.stem for p in FLOW_FILES])
def flow_path(request) -> Path:
    return request.param


@pytest.fixture
def flow(flow_path) -> dict:
    return json.loads(flow_path.read_text(encoding="utf-8"))


@pytest.fixture
def synthetic() -> dict:
    return json.loads(SYNTHETIC.read_text(encoding="utf-8"))


class FakeJudge:
    """Offline stand-in for OpenAIJudge. Decides by simple text rules."""

    model = "fake-judge"

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.calls = []

    def review_fields(self, node, fields, deterministic_findings):
        self.calls.append({"node": node, "fields": fields, "context": deterministic_findings})
        out = []
        for f in fields:
            if "up to two nudges" in f["after"]:
                cls = "meaning_changed"
                rules = ["exactly one nudge"]
            else:
                cls = "rewording_same_meaning"
                rules = []
            out.append({"field_id": f["field_id"], "classification": cls, "confidence": self.confidence,
                        "evidence": ["before: ...", "after: ..."], "affected_rules": rules})
        return out

    def summarize(self, digest):
        return [f"Verdict {digest['verdict']} with {len(digest['findings'])} findings."]


@pytest.fixture
def fake_judge() -> FakeJudge:
    return FakeJudge()
