"""CLI exit codes and report files."""

import json

import pytest

import mutations as mut
from flow_guard.cli import main


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def check(tmp_path, a, b, *extra):
    return main(["check", a, b, "--report-dir", str(tmp_path / "out"), *extra])


def test_exit_codes(tmp_path, synthetic):
    src = _write(tmp_path, "in.json", synthetic)
    assert check(tmp_path, src, src) == 0
    warn = _write(tmp_path, "warn.json", mut.reorder_edges(synthetic))
    assert check(tmp_path, src, warn) == 1
    assert check(tmp_path, src, warn, "--fail-on-warning") == 2
    assert check(tmp_path, src, _write(tmp_path, "fail.json", mut.delete_edge(synthetic))) == 2
    renamed = _write(tmp_path, "renamed.json", mut.rename_ids(synthetic))
    assert check(tmp_path, src, renamed) == 1
    assert check(tmp_path, src, renamed, "--strict-ids") == 2
    unknown = _write(tmp_path, "unknown.json", mut.add_unknown_key(synthetic))
    assert check(tmp_path, src, unknown, "--strict-schema") == 2


def test_usage_and_io_errors_exit_3(tmp_path, synthetic):
    src = _write(tmp_path, "in.json", synthetic)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert check(tmp_path, src, str(bad)) == 3
    missing = _write(tmp_path, "missing.json", {"data": {"prompt": "hi"}})
    assert check(tmp_path, src, missing) == 3
    assert check(tmp_path, src, str(tmp_path / "does-not-exist.json")) == 3
    policy = tmp_path / "p.json"
    policy.write_text('{"severities": {"nope": "INFO"}}')
    assert check(tmp_path, src, src, "--policy", str(policy)) == 3
    with pytest.raises(SystemExit) as info:
        main(["check", src])
    assert info.value.code == 3


def test_validate_command(tmp_path, synthetic, capsys):
    assert main(["validate", _write(tmp_path, "in.json", synthetic)]) == 0
    dangling = mut.rename_ids_leave_dangling(synthetic)
    assert main(["validate", _write(tmp_path, "d.json", dangling)]) == 2
    assert "dangling_reference" in capsys.readouterr().out


def test_reports_are_written_and_stable(tmp_path, synthetic):
    src = _write(tmp_path, "in.json", synthetic)
    out = _write(tmp_path, "out.json", mut.swap_edge_destinations(mut.reword_instruction(synthetic)))
    check(tmp_path, src, out, "--no-llm")
    first = (tmp_path / "out" / "report.json").read_text(encoding="utf-8")
    check(tmp_path, src, out, "--no-llm")
    assert (tmp_path / "out" / "report.json").read_text(encoding="utf-8") == first

    data = json.loads(first)
    assert data["schema_version"] and data["verdict"] == "FAIL"
    assert set(data["counts"]) == {"CRITICAL", "WARNING", "INFO"}
    assert len(data["node_mapping"]) == 7
    keys = {"id", "severity", "category", "node_id", "path", "message", "before", "after", "source", "unified_diff"}
    assert all(keys <= set(f) for f in data["findings"])
    assert data["judge"]["status"] == "skipped: --no-llm"

    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "❌ FAIL" in md
    assert "## Node mapping" in md and "## Findings" in md
    assert "Transition now leads somewhere else" in md
    assert "```mermaid" in md and "CHANGED" in md
    assert "AI judge was not run" in md


def test_identity_report_says_pass(tmp_path, synthetic):
    src = _write(tmp_path, "in.json", synthetic)
    check(tmp_path, src, src)
    data = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS" and data["findings"] == []
    assert data["judge"]["status"].startswith("skipped")


def test_input_files_are_not_modified(tmp_path, synthetic):
    src = _write(tmp_path, "in.json", synthetic)
    out = _write(tmp_path, "out.json", mut.nodes_to_dict(synthetic))
    before = (tmp_path / "in.json").read_bytes(), (tmp_path / "out.json").read_bytes()
    check(tmp_path, src, out)
    assert ((tmp_path / "in.json").read_bytes(), (tmp_path / "out.json").read_bytes()) == before
