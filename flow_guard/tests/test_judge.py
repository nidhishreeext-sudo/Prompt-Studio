"""OpenAIJudge plumbing with a fake client, plus one live smoke test."""

import json
import os
from types import SimpleNamespace

import httpx
import openai
import pytest

import mutations as mut
from flow_guard import OpenAIJudge, Verdict, compare
from flow_guard.judge import NODE_SCHEMA, judge_from_env


def _completion(payload: dict):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


class FakeClient:
    def __init__(self, responder, reject_temperature=False, transient_failures=0):
        self.requests = []
        self.responder = responder
        self.reject_temperature = reject_temperature
        self.transient_failures = transient_failures
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        request = httpx.Request("POST", "https://api.invalid/v1/chat/completions")
        if self.reject_temperature and "temperature" in kwargs:
            raise openai.BadRequestError("Unsupported value: 'temperature' does not support 0",
                                         response=httpx.Response(400, request=request), body=None)
        if self.transient_failures:
            self.transient_failures -= 1
            raise openai.APITimeoutError(request=request)
        return _completion(self.responder(kwargs))


def field_responder(classification):
    def respond(kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "diff_summary":
            return {"bullets": ["One instruction was reworded."]}
        payload = json.loads(kwargs["messages"][1]["content"])
        return {"fields": [{"field_id": f["field_id"], "classification": classification, "confidence": 0.9,
                            "evidence": ["before: x", "after: y"], "affected_rules": []} for f in payload["fields"]]}
    return respond


def test_openai_judge_request_shape_and_cache(tmp_path, synthetic):
    client = FakeClient(field_responder("rewording_same_meaning"))
    judge = OpenAIJudge(model="test-model", api_key="sk-secret-123", cache_dir=tmp_path, client=client)
    out = mut.reword_instruction(synthetic)

    report = compare(synthetic, out, judge=judge)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    assert "rewording_same_meaning" in report.categories()
    assert report.judge_summary == ["One instruction was reworded."]
    first_calls = len(client.requests)
    assert first_calls == 2  # one node batch + one summary

    req = client.requests[0]
    assert req["model"] == "test-model" and req["temperature"] == 0
    assert req["response_format"]["json_schema"]["strict"] is True
    assert req["response_format"]["json_schema"]["schema"] == NODE_SCHEMA
    assert "DATA" in req["messages"][0]["content"]

    again = compare(synthetic, out, judge=judge)  # served from the disk cache
    assert len(client.requests) == first_calls
    assert [f.to_dict() for f in again.findings] == [f.to_dict() for f in report.findings]
    assert "sk-secret-123" not in repr(judge)
    for path in tmp_path.iterdir():
        assert "sk-secret-123" not in path.read_text(encoding="utf-8")


def test_openai_judge_retries_without_temperature(tmp_path, synthetic):
    client = FakeClient(field_responder("meaning_changed"), reject_temperature=True)
    judge = OpenAIJudge(model="m", api_key="k", cache_dir=tmp_path, client=client)
    report = compare(synthetic, mut.reword_instruction(synthetic), judge=judge, summarize=False)
    assert "meaning_changed" in report.categories()
    assert "temperature" in client.requests[0] and "temperature" not in client.requests[-1]


def test_openai_judge_retries_transient_errors(tmp_path, synthetic, monkeypatch):
    monkeypatch.setattr("flow_guard.judge.time.sleep", lambda s: None)
    client = FakeClient(field_responder("rewording_same_meaning"), transient_failures=2)
    judge = OpenAIJudge(model="m", api_key="k", cache_dir=tmp_path, client=client, max_retries=3)
    report = compare(synthetic, mut.reword_instruction(synthetic), judge=judge, summarize=False)
    assert "rewording_same_meaning" in report.categories()
    assert len(client.requests) == 3


def test_judge_only_sees_rule_undecidable_fields(tmp_path, synthetic):
    client = FakeClient(field_responder("identical_meaning"))
    judge = OpenAIJudge(model="m", api_key="k", cache_dir=tmp_path, client=client)
    report = compare(synthetic, mut.truncate_instruction(synthetic), judge=judge)
    assert client.requests == [] or all(
        r["response_format"]["json_schema"]["name"] == "diff_summary" for r in client.requests)
    assert report.verdict is Verdict.FAIL


def test_api_key_never_reaches_cache_or_reports(tmp_path, synthetic, monkeypatch):
    """The key must not appear in the cache dir, report.json or report.md, even
    when an API error message echoes it back."""
    from flow_guard.report import write_reports

    # Built at runtime so secret scanners don't flag this file.
    fake_key = "sk-" + "proj-" + "FAKE0TESTKEY" * 4
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    cache_dir, report_dir = tmp_path / "cache", tmp_path / "reports"

    # 1. Normal run: judge calls + summary, then cached rerun.
    client = FakeClient(field_responder("meaning_changed"))
    judge = OpenAIJudge(model="m", cache_dir=cache_dir, client=client)  # key comes from the env
    out = mut.nudge_meaning_change(mut.reword_instruction(synthetic))
    report = compare(synthetic, out, judge=judge)
    assert client.requests, "the fake client must have been called"
    write_reports(report, report_dir)
    compare(synthetic, out, judge=judge)

    # 2. Failing run: the API error text contains the key.
    def leaky(kwargs):
        raise AssertionError("unreachable")

    leaky_client = FakeClient(leaky)

    def create(**kwargs):
        request = httpx.Request("POST", "https://api.invalid/v1/chat/completions")
        raise openai.AuthenticationError(f"Incorrect API key provided: {fake_key}",
                                         response=httpx.Response(401, request=request), body=None)

    leaky_client.chat.completions.create = create
    failing = compare(synthetic, mut.reword_instruction(synthetic),
                      judge=OpenAIJudge(model="m2", cache_dir=cache_dir, client=leaky_client))
    assert failing.judge_status.startswith("partial")
    write_reports(failing, tmp_path / "reports_failed")

    assert fake_key not in repr(judge)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.parent == cache_dir for p in files) and any(p.name == "report.md" for p in files)
    for path in files:
        assert fake_key.encode() not in path.read_bytes(), f"API key leaked into {path.relative_to(tmp_path)}"


def test_no_key_means_no_judge(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge, status = judge_from_env()
    assert judge is None and "OPENAI_API_KEY" in status


def test_default_model_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "my-model")
    assert OpenAIJudge(api_key="k").model == "my-model"
    monkeypatch.delenv("OPENAI_JUDGE_MODEL")
    assert OpenAIJudge(api_key="k").model == "gpt-6-luna"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_judge_flags_nudge_change(tmp_path, synthetic):
    judge = OpenAIJudge(cache_dir=tmp_path)
    report = compare(synthetic, mut.nudge_meaning_change(synthetic), judge=judge)
    assert report.judge_status == "enabled", report.judge_status
    llm = [f for f in report.findings if f.source == "llm"]
    assert llm and llm[0].category in ("meaning_changed", "needs_human_review")
