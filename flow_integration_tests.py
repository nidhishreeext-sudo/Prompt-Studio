"""
flow_integration_tests.py — standing regression suite for the flow-JSON output
path: the output flow builder (flow_output.py), the flow_guard integrity check
(flow_check.py) and their wiring into /api/generate.

Run directly: `python flow_integration_tests.py`. Exits 0 if every check
passes, 1 if any fails. Every Gemini call is mocked; no API key is needed.

`--skip-install` skips the slow clean-venv install check (group 10).
"""

import copy
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "unit-test-placeholder-key")
os.environ.pop("OPENAI_API_KEY", None)  # the integrity check must never need it

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "flow_guard" / "tests" / "fixtures" / "synthetic_flow.json"
sys.path.insert(0, str(ROOT / "flow_guard" / "tests"))  # flow_guard's own mutation helpers

import main  # noqa: E402
import app as flask_app_module  # noqa: E402
import flow_check  # noqa: E402
import flow_output  # noqa: E402
import flow_guard  # noqa: E402
import mutations  # noqa: E402

FAILURES = []
STRUCTURAL = {
    "node_removed", "node_added", "node_type_changed", "node_id_shift", "start_node_changed",
    "edge_removed", "edge_added", "edge_redirected", "condition_changed", "condition_emptied",
    "go_back_to_previous_changed", "topology_not_equivalent", "switch_expression_changed",
    "switch_case_changed", "switch_default_changed", "global_variable_changed", "supported_modes_changed",
    "allow_go_back_changed", "go_back_edges_changed", "jump_to_edges_changed", "global_node_changed",
    "overlay_language_removed", "prompt_key_removed", "dangling_reference", "unreachable_node",
}


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)
    return condition


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = [type("C", (), {"finish_reason": "STOP"})()]


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def node(flow: dict, node_id: str) -> dict:
    return next(n for n in flow["data"]["conversationNodes"]["nodes"] if n["id"] == node_id)


class FakeGemini:
    """Routes each mocked Gemini call by the system prompt it carries.
    `extractor(raw_text) -> str` decides what the distill step returns."""

    def __init__(self, extractor=None):
        self.extractor = extractor or (lambda text: text)
        self.extract_calls = []
        self.total_calls = 0

    def __call__(self, model, contents, config=None):
        self.total_calls += 1
        if "You are a prompt-cleaning tool" in contents:
            raw = contents.split("---RAW PROMPT---\n", 1)[1]
            self.extract_calls.append(raw)
            return FakeResponse(self.extractor(raw))
        if "explicitly restricts which spoken languages" in contents:
            return FakeResponse("NONE")
        if "CUSTOM LANGUAGE / SPEECH RULES" in contents:
            return FakeResponse("")
        if "reviewing a generated language-specific voice prompt" in contents:
            return FakeResponse("Looks fine, no issues found.")
        return FakeResponse("### Colloquial Speech\nSpeak naturally and warmly, confirming the booking details.")


def post(raw_prompt: str, mode: str, languages=("Kannada", "Tamil"), fake=None, extra=None):
    fake = fake or FakeGemini()
    original = main.client.models.generate_content
    main.client.models.generate_content = fake
    try:
        client = flask_app_module.app.test_client()
        resp = client.post("/api/generate", json={
            "raw_prompt": raw_prompt, "mode": mode, "languages": list(languages),
            "model": main.DEFAULT_MODEL, "user_name": "flow-integration-suite", **(extra or {}),
        })
        return resp, resp.get_json(), fake
    finally:
        main.client.models.generate_content = original


def post_flow(flow: dict, mode: str, **kw):
    return post(json.dumps(flow, ensure_ascii=False), mode, **kw)


def structure_findings(fc: dict) -> list:
    return [f for f in fc.get("findings", []) if f["category"] in STRUCTURAL]


def strip_sentence(sentence: str):
    return lambda text: text.replace(sentence, "")


# ---------------------------------------------------------------------------

def test_d3_key_convention():
    print("\n--- 0. D3 language-prompt key convention ---")
    keys = [k for k in load_fixture()["data"]["prompts"] if k != "default"]
    ok = check("fixture data.prompts uses lang:<ll>-IN keys", bool(keys) and all(re.fullmatch(r"lang:[a-z]{2}-IN", k) for k in keys),
               detail=str(keys))
    ok &= check("helper maps HI to the fixture's own key", flow_output.language_prompt_key("HI") == "lang:hi-IN")
    ok &= check("helper maps TE to the fixture's own key", flow_output.language_prompt_key("TE") == "lang:te-IN")
    return ok


def test_scoped_mode():
    print("\n--- 1. Scoped mode on the sample flow ---")
    fx = load_fixture()
    resp, data, _ = post_flow(fx, "scoped")
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    fc, out = data["flow_check"], data["flow_json"]
    ok &= check("flow_check ran without error", "error" not in fc, detail=str(fc.get("error")))
    stripped = copy.deepcopy(out)
    for key in ("lang:kn-IN", "lang:ta-IN"):
        stripped["data"]["prompts"].pop(key, None)
    ok &= check("output equals input apart from the attached language prompts", stripped == fx)
    ok &= check("topology_equivalent is true", fc.get("topology_equivalent") is True)
    ok &= check("verdict is PASS or PASS_WITH_WARNINGS", fc.get("verdict") in ("PASS", "PASS_WITH_WARNINGS"), detail=fc.get("verdict"))
    ok &= check("every warning is an expected overlay addition",
                all(f.get("expected") and f["category"] == "prompt_key_added" for f in fc["findings"]),
                detail=str([(f["category"], f["location"]["path"]) for f in fc["findings"]]))
    ok &= check("attached prompts sit under data.prompts with lang:<tag> keys",
                out["data"]["prompts"].get("lang:kn-IN") == data["language_prompts"]["KN"]
                and out["data"]["prompts"].get("lang:ta-IN") == data["language_prompts"]["TA"])
    return ok


def test_extract_strips_time_rule():
    print("\n--- 2. Extract mode: a language rule with a time example is stripped ---")
    fx = load_fixture()
    rule = "When you say a time, say 10:30 AM as 'saadhe das baje'. "
    greet = node(fx, "node-greet")
    greet["instruction"] = greet["instruction"].replace("Speak in short sentences. ", "Speak in short sentences. " + rule)
    resp, data, fake = post_flow(fx, "extract", fake=FakeGemini(strip_sentence(rule)))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    fc = data["flow_check"]
    ok &= check("no structural findings", not structure_findings(fc), detail=str(structure_findings(fc)))
    ok &= check("raw_verdict (flow_guard's own) is FAIL", fc.get("raw_verdict") == "FAIL", detail=fc.get("raw_verdict"))
    ok &= check("adjusted verdict is not FAIL", fc.get("verdict") != "FAIL", detail=fc.get("verdict"))
    down = fc.get("downgraded") or []
    ok &= check("exactly one finding was re-labelled", len(down) == 1, detail=str(down))
    if down:
        tokens = down[0]["tokens"]
        ok &= check("the re-labelled finding is the 10:30 AM time on the rewritten node",
                    down[0]["path"] == "nodes[node-greet].instruction"
                    and [(t["kind"], t["token"]) for t in tokens] == [("time", "10:30 AM")], detail=str(tokens))
        ok &= check("the removed figure is quoted with its surrounding text",
                    any("saadhe das baje" in c for t in tokens for c in t["contexts"]), detail=str(tokens))
    out_greet = node(data["flow_json"], "node-greet")["instruction"]
    ok &= check("output node carries the distilled instruction", rule not in out_greet and "Lakshmi" in out_greet)
    return ok


def test_extract_drops_placeholder():
    print("\n--- 3. Extract mode: a {{variable}} placeholder is dropped ---")
    fx = load_fixture()
    sentence = "Always address the devotee as {{devotee_name}} ji. "
    resp, data, _ = post_flow(fx, "extract", fake=FakeGemini(strip_sentence(sentence)))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    fc = data["flow_check"]
    ok &= check("verdict is FAIL", fc.get("verdict") == "FAIL", detail=fc.get("verdict"))
    lost = [f for f in fc["findings"] if f["category"] == "token_lost" and f["location"]["path"] == "data.prompt"]
    ok &= check("the placeholder loss stays CRITICAL", bool(lost) and all(f["severity"] == "CRITICAL" for f in lost), detail=str(lost))
    ok &= check("nothing was downgraded", fc.get("downgraded") == [], detail=str(fc.get("downgraded")))
    return ok


def test_extract_fallback():
    print("\n--- 4. Extractor returns empty / over-stripped text ---")
    fx = load_fixture()
    greet, pooja = node(fx, "node-greet")["instruction"], node(fx, "node-pooja")["instruction"]

    def extractor(text):
        if "Greet" in text and "nudge" in text:
            return ""                      # e.g. a safety-blocked response
        if text.strip() == pooja.strip():
            return text[: len(text) // 10]  # over-summarized
        return text
    resp, data, _ = post_flow(fx, "extract", fake=FakeGemini(extractor))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    out = data["flow_json"]
    ok &= check("empty extraction: original instruction kept byte-for-byte (escapes included)",
                node(out, "node-greet")["instruction"] == greet)
    ok &= check("over-stripped extraction: original instruction kept byte-for-byte",
                node(out, "node-pooja")["instruction"] == pooja)
    notes = {n["field"]: n for n in data["flow_field_notes"]}
    ok &= check("field note for the empty extraction",
                notes.get("nodes[node-greet].instruction", {}).get("status") == "kept_original"
                and "nothing" in notes["nodes[node-greet].instruction"]["detail"])
    ok &= check("field note for the over-stripped extraction",
                notes.get("nodes[node-pooja].instruction", {}).get("status") == "kept_original"
                and "40%" in notes["nodes[node-pooja].instruction"]["detail"])
    out_nodes = out["data"]["conversationNodes"]["nodes"]
    fields = flow_output.list_rewritable_fields(load_fixture())
    ok &= check("no rewritable instruction is empty in the output",
                all(out_nodes[f.node_key]["instruction"].strip() for f in fields if f.node_key is not None))
    return ok


def test_extraction_preamble_stripped():
    print("\n--- 4b. A stray extraction preamble never lands in the output flow ---")
    fx = load_fixture()
    preamble = "Here is the cleaned prompt:\n"
    resp, data, _ = post_flow(fx, "extract", fake=FakeGemini(
        lambda text: preamble + text.replace("Speak in short sentences. ", "")))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    out = data["flow_json"]
    ok &= check("the node rewritten with a preamble was still accepted",
                any(n["field"] == "nodes[node-greet].instruction" and n["status"] == "rewritten"
                    for n in data["flow_field_notes"]))
    ok &= check("no preamble anywhere in the output flow", "Here is the cleaned prompt" not in json.dumps(out))
    ok &= check("no preamble in business_logic", "Here is the cleaned prompt" not in data["business_logic"])
    ok &= check("rewritten instruction starts with its real content",
                node(out, "node-greet")["instruction"].startswith("You are Lakshmi"))
    return ok


def remove_node(doc: dict) -> dict:
    """mutations.py has no node-removal helper; this drops one non-start node."""
    out = copy.deepcopy(doc)
    conv = out["data"]["conversationNodes"]
    conv["nodes"] = [n for n in conv["nodes"] if n["id"] != "node-callback"]
    return out


def test_mutations_caught():
    print("\n--- 5. Corrupting the produced output flow is caught ---")
    fx = load_fixture()
    rule = "Speak in short sentences. "
    resp, data, _ = post_flow(fx, "extract", fake=FakeGemini(strip_sentence(rule)))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    out = data["flow_json"]
    rewritten = [n["field"] for n in data["flow_field_notes"] if n["status"] in ("rewritten", "replaced")]
    attached = [n["field"].split("data.prompts.", 1)[1] for n in data["flow_field_notes"] if n["status"] == "attached"]
    base = flow_check.run_flow_check(fx, out, rewritten, attached)
    ok &= check("unmutated output is not FAIL", base.get("verdict") != "FAIL", detail=str(base.get("verdict")))
    cases = [
        ("remove an edge", mutations.delete_edge),
        ("change a condition", mutations.change_condition_word),
        ("redirect an edge", mutations.swap_edge_destinations),
        ("remove a node", remove_node),
        ("change a node type", mutations.change_node_type),
    ]
    for label, mutate in cases:
        fc = flow_check.run_flow_check(fx, mutate(out), rewritten, attached)
        ok &= check(f"{label}: verdict FAIL", fc.get("verdict") == "FAIL",
                    detail=str(fc.get("verdict") or fc.get("error")))
    return ok


def test_input_not_mutated():
    print("\n--- 6. The input flow is never mutated ---")
    fx = load_fixture()
    node(fx, "node-followup")["name"] = "Hindi"
    before = json.dumps(fx, ensure_ascii=False, sort_keys=False)
    snapshot = copy.deepcopy(fx)
    distillation = flow_output.distill_flow(fx, model=main.DEFAULT_MODEL,
                                            extract_fn=lambda text, model: text.replace("Speak in short sentences. ", ""))
    built = flow_output.build_output_flow(fx, distillation, {"KN": "ಕನ್ನಡ ಪ್ರಾಂಪ್ಟ್", "HI": "हिंदी प्रॉम्प्ट"})
    flow_check.run_flow_check(fx, built.flow, built.rewritten_paths, built.attached_keys, built.created_prompts)
    ok = check("input deep-equal after distill, build and check", fx == snapshot)
    ok &= check("input byte-identical after distill, build and check", json.dumps(fx, ensure_ascii=False) == before)
    return ok


def test_language_nodes_untouched():
    print("\n--- 7. Language-named nodes are left untouched ---")
    fx = load_fixture()
    lang_node = node(fx, "node-followup")
    lang_node["name"] = "Hindi"
    original = lang_node["instruction"]
    marker = "Business fact kept from the Hindi node."

    def extractor(text):
        return marker if text.strip() == original.strip() else text
    resp, data, fake = post_flow(fx, "extract", fake=FakeGemini(extractor))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    ok &= check("language node instruction is byte-identical in the output",
                node(data["flow_json"], "node-followup")["instruction"] == original)
    note = next((n for n in data["flow_field_notes"] if n["field"] == "nodes[node-followup].instruction"), {})
    ok &= check("field note says the language node was left untouched", note.get("status") == "language_node_untouched")
    ok &= check("business_logic uses the distilled language-node text, not its raw rules",
                marker in data["business_logic"] and original.strip() not in data["business_logic"])
    _, scoped, _ = post_flow(fx, "scoped")
    ok &= check("scoped mode also notes the language node",
                any(n["status"] == "language_node_untouched" for n in scoped["flow_field_notes"]))
    return ok


def test_failed_language_not_attached():
    print("\n--- 7b. A language that ended in an error state is not attached ---")
    fx = load_fixture()
    built = flow_output.build_output_flow(fx, None, {
        "KN": flow_output.GENERATION_FAILED_PREFIX + " KN.\n\nEvery generation attempt ...",
        "TA": "தமிழ் ப்ராம்ப்ட்",
    })
    prompts = built.flow["data"]["prompts"]
    ok = check("failed language not attached", "lang:kn-IN" not in prompts)
    ok &= check("successful language attached", prompts.get("lang:ta-IN") == "தமிழ் ப்ராம்ப்ட்")
    ok &= check("field note explains the missing language",
                any(n["status"] == "not_attached" and "kn-IN" in n["field"] for n in built.notes))
    return ok


def test_guard_crash():
    print("\n--- 8. Guard crash never breaks /api/generate ---")
    original = flow_guard.compare

    def boom(*args, **kwargs):
        raise RuntimeError("simulated guard crash")
    flow_guard.compare = boom
    try:
        resp, data, _ = post_flow(load_fixture(), "scoped")
    finally:
        flow_guard.compare = original
    ok = check("request still succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    ok &= check("language prompts still returned", set(data.get("language_prompts", {})) == {"KN", "TA"})
    ok &= check("flow_check.error is set", "simulated guard crash" in (data["flow_check"].get("error") or ""),
                detail=str(data["flow_check"]))
    ok &= check("flow_json still returned", isinstance(data.get("flow_json"), dict))
    return ok


def test_no_key_no_network():
    print("\n--- 8b. Integrity check runs with no OPENAI_API_KEY and no network ---")
    fx = load_fixture()
    original_connect = socket.socket.connect

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")
    socket.socket.connect = no_network
    try:
        fc = flow_check.run_flow_check(fx, copy.deepcopy(fx))
    finally:
        socket.socket.connect = original_connect
    ok = check("OPENAI_API_KEY is unset", "OPENAI_API_KEY" not in os.environ)
    ok &= check("identical flows give PASS with no network", fc.get("verdict") == "PASS", detail=str(fc.get("error")))
    ok &= check("judge reported as skipped", (fc.get("judge") or "").startswith("skipped"))
    return ok


def _run(cmd, cwd=ROOT, env=None, timeout=900):
    env = dict(env or os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def test_plain_text_unchanged():
    print("\n--- 9. Plain-text input behaves exactly as before ---")
    resp, data, _ = post("You are a booking assistant. Collect the caller's name and confirm the slot.", "extract")
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    ok &= check("response keys are exactly the pre-existing ones",
                set(data) == {"business_logic", "language_prompts", "warnings", "model", "generation_id"},
                detail=str(sorted(data)))
    ok &= check("no flow keys present (absent, not null)",
                not any(k in data for k in ("flow_json", "flow_check", "flow_field_notes")))

    guardrails = _run([sys.executable, "guardrail_regression_tests.py"])
    ok &= check("guardrail_regression_tests.py passes unchanged: all 8 groups, 44 checks",
                guardrails.returncode == 0 and "all 8 check groups PASSED" in guardrails.stdout
                and guardrails.stdout.count("[PASS]") == 44 and "[FAIL]" not in guardrails.stdout,
                detail=guardrails.stdout[-400:] + guardrails.stderr[-400:])
    return ok


def test_clean_install(skip: bool):
    print("\n--- 10. Clean install: fresh venv, requirements.txt, app boots ---")
    if skip:
        print("[SKIP] --skip-install given")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        made = _run([sys.executable, "-m", "venv", str(venv)])
        if not check("venv created", made.returncode == 0, detail=made.stderr[-400:]):
            return False
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pip = _run([str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        ok = check("pip install -r requirements.txt from the repo root", pip.returncode == 0, detail=pip.stderr[-600:])
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        env["GEMINI_API_KEY"] = "unit-test-placeholder-key"
        probe = (
            "import flow_guard, app\n"
            "assert flow_guard.__file__ and 'site-packages' in flow_guard.__file__, flow_guard.__file__\n"
            "from flow_guard import compare\n"
            "r = app.app.test_client().get('/api/models')\n"
            "assert r.status_code == 200 and r.get_json()['models'], r.status_code\n"
            "print('BOOT_OK', flow_guard.__file__)\n"
        )
        boot = _run([str(py), "-c", probe], env=env)  # cwd = repo root, like `gunicorn app:app`
        ok &= check("import app, flow_guard from site-packages, /api/models responds, no OPENAI_API_KEY",
                    boot.returncode == 0 and "BOOT_OK" in boot.stdout, detail=(boot.stdout + boot.stderr)[-600:])
    return ok


def test_flow_guard_pytest():
    print("\n--- 11. flow_guard's own pytest ---")
    res = _run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT / "flow_guard")
    tail = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-300:]
    return check("flow_guard pytest passes", res.returncode == 0 and "failed" not in tail and "error" not in tail,
                 detail=tail)


FLOW_KEYS = ("flow_json", "flow_check", "flow_field_notes", "flow_language_keys")
PLAIN_KEYS = {"business_logic", "language_prompts", "warnings", "model", "generation_id"}
TIME_RULE = "When you say a time, say 10:30 AM as 'saadhe das baje'. "


def time_rule_flow() -> dict:
    fx = load_fixture()
    greet = node(fx, "node-greet")
    greet["instruction"] = greet["instruction"].replace("Speak in short sentences. ", "Speak in short sentences. " + TIME_RULE)
    return fx


class CountingCompare:
    def __init__(self):
        self.calls = 0
        self.original = flow_guard.compare

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.original(*args, **kwargs)

    def __enter__(self):
        flow_guard.compare = self
        return self

    def __exit__(self, *exc):
        flow_guard.compare = self.original


def test_output_format_prompt():
    print("\n--- 12. output_format=\"prompt\": prompts only, same text as flow_json mode ---")
    ok = True
    for mode in ("extract", "scoped"):
        with CountingCompare() as counter:
            resp_p, prompt_run, _ = post_flow(time_rule_flow(), mode, fake=FakeGemini(strip_sentence(TIME_RULE)),
                                              extra={"output_format": "prompt"})
        ok &= check(f"{mode}: request succeeded", resp_p.status_code == 200, detail=str(prompt_run)[:300])
        ok &= check(f"{mode}: no flow keys at all (absent, not null)", not any(k in prompt_run for k in FLOW_KEYS),
                    detail=str(sorted(prompt_run)))
        ok &= check(f"{mode}: response keys are exactly the plain ones", set(prompt_run) == PLAIN_KEYS, detail=str(sorted(prompt_run)))
        ok &= check(f"{mode}: flow_guard compare never called", counter.calls == 0, detail=str(counter.calls))
        with CountingCompare() as counter:
            _, flow_run, _ = post_flow(time_rule_flow(), mode, fake=FakeGemini(strip_sentence(TIME_RULE)),
                                       extra={"output_format": "flow_json"})
        ok &= check(f"{mode}: flow_json mode does call compare (sanity)", counter.calls == 1, detail=str(counter.calls))
        ok &= check(f"{mode}: business_logic byte-identical in both modes",
                    prompt_run["business_logic"] == flow_run["business_logic"])
        ok &= check(f"{mode}: every language prompt byte-identical in both modes",
                    prompt_run["language_prompts"] == flow_run["language_prompts"] and set(prompt_run["language_prompts"]) == {"KN", "TA"})
        ok &= check(f"{mode}: warnings identical in both modes", prompt_run["warnings"] == flow_run["warnings"])
    return ok


def test_output_format_default():
    print("\n--- 13. output_format=\"flow_json\" and a missing output_format give today's response ---")
    _, explicit, _ = post_flow(time_rule_flow(), "extract", fake=FakeGemini(strip_sentence(TIME_RULE)),
                               extra={"output_format": "flow_json"})
    _, default, _ = post_flow(time_rule_flow(), "extract", fake=FakeGemini(strip_sentence(TIME_RULE)))
    ok = check("explicit flow_json and missing output_format are identical", explicit == default)
    ok &= check("response keys are today's plus flow_language_keys",
                set(default) == PLAIN_KEYS | set(FLOW_KEYS), detail=str(sorted(default)))
    fc = default["flow_check"]
    ok &= check("flow_check keeps every existing key, adds only 'languages'",
                set(fc) == {"verdict", "raw_verdict", "counts", "raw_counts", "topology_equivalent", "findings",
                            "preexisting_problems", "downgraded", "judge", "raw_report", "languages"}, detail=str(sorted(fc)))
    ok &= check("verdicts unchanged from the existing behavior", fc["verdict"] == "PASS_WITH_WARNINGS" and fc["raw_verdict"] == "FAIL")
    ok &= check("language summary: all requested languages attached",
                fc["languages"] == {"requested": ["KN", "TA"], "attached": ["KN", "TA"], "missing": []}, detail=str(fc["languages"]))
    return ok


def test_output_format_invalid():
    print("\n--- 14. An invalid output_format is rejected ---")
    resp, data, fake = post_flow(load_fixture(), "extract", extra={"output_format": "pdf"})
    ok = check("returns 400", resp.status_code == 400, detail=str(resp.status_code))
    ok &= check("clear message naming the field and the valid values",
                data.get("error") == "Unknown output_format 'pdf' (use one of: prompt, flow_json)", detail=str(data))
    ok &= check("rejected before any Gemini call", fake.total_calls == 0, detail=str(fake.total_calls))
    return ok


def test_output_format_plain_text():
    print("\n--- 15. Plain-text input ignores output_format ---")
    text = "You are a booking assistant. Collect the caller's name and confirm the slot."
    _, baseline, _ = post(text, "extract")
    ok = True
    for value in ("flow_json", "prompt", "pdf"):
        resp, data, _ = post(text, "extract", extra={"output_format": value})
        ok &= check(f"output_format={value!r}: 200 and the same response as without it",
                    resp.status_code == 200 and data == baseline, detail=str(data)[:300])
    ok &= check("no flow keys in the plain-text response", set(baseline) == PLAIN_KEYS, detail=str(sorted(baseline)))
    return ok


def test_failed_language_reported():
    print("\n--- 16. A language that ends in an error state is reported missing ---")
    original = main._generate_one_language

    def fail_kannada(*args):
        if args[1] == "KN":
            return "KN", flow_output.GENERATION_FAILED_PREFIX + " KN.\n\nEvery generation attempt failed.", [
                "Generation failed for 'KN': output remained in the wrong language."]
        return original(*args)
    main._generate_one_language = fail_kannada
    try:
        resp, data, _ = post_flow(load_fixture(), "extract")
    finally:
        main._generate_one_language = original
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    langs = data["flow_check"]["languages"]
    ok &= check("summary lists KN as missing because generation failed",
                langs == {"requested": ["KN", "TA"], "attached": ["TA"],
                          "missing": [{"language": "KN", "reason": "generation failed"}]}, detail=str(langs))
    ok &= check("KN overlay is not in the flow", "lang:kn-IN" not in data["flow_json"]["data"]["prompts"])
    ok &= check("KN is not in flow_language_keys", data["flow_language_keys"] == {"TA": "lang:ta-IN"},
                detail=str(data["flow_language_keys"]))
    return ok


def test_flow_language_keys():
    print("\n--- 17. flow_language_keys matches the overlay keys in the flow ---")
    fx = load_fixture()
    resp, data, _ = post_flow(fx, "extract", languages=("Kannada", "Tamil", "English", "Hindi"))
    ok = check("request succeeded", resp.status_code == 200, detail=str(data)[:300])
    if not ok:
        return ok
    keys, prompts = data["flow_language_keys"], data["flow_json"]["data"]["prompts"]
    ok &= check("each key comes from language_prompt_key()",
                keys == {lang: flow_output.language_prompt_key(lang) for lang in ("KN", "TA", "EN", "HI")}, detail=str(keys))
    ok &= check("every mapped key is present in data.prompts with that language's text",
                all(prompts.get(key) == data["language_prompts"][lang] for lang, key in keys.items()))
    new_keys = set(prompts) - set(fx["data"]["prompts"])
    ok &= check("keys added to data.prompts are exactly the mapped ones (lang:hi-IN already existed and was replaced)",
                new_keys | {"lang:hi-IN"} == set(keys.values()), detail=str(sorted(new_keys)))
    return ok


def test_call_count():
    print("\n--- Gemini extract calls for the sample flow ---")
    fake = FakeGemini()
    post_flow(load_fixture(), "extract", fake=fake)
    expected = len(flow_output.list_rewritable_fields(load_fixture())) + len(flow_output.list_language_node_fields(load_fixture()))
    print(f"  extract calls: {len(fake.extract_calls)} (one per field); all Gemini calls in the request: {fake.total_calls}")
    return check("one extract call per distillable field", len(fake.extract_calls) == expected == 5,
                 detail=f"{len(fake.extract_calls)} calls, expected {expected}")


def main_entry():
    skip_install = "--skip-install" in sys.argv
    results = [
        test_d3_key_convention(),
        test_scoped_mode(),
        test_extract_strips_time_rule(),
        test_extract_drops_placeholder(),
        test_extract_fallback(),
        test_extraction_preamble_stripped(),
        test_mutations_caught(),
        test_input_not_mutated(),
        test_language_nodes_untouched(),
        test_failed_language_not_attached(),
        test_guard_crash(),
        test_no_key_no_network(),
        test_plain_text_unchanged(),
        test_clean_install(skip_install),
        test_flow_guard_pytest(),
        test_call_count(),
        test_output_format_prompt(),
        test_output_format_default(),
        test_output_format_invalid(),
        test_output_format_plain_text(),
        test_failed_language_reported(),
        test_flow_language_keys(),
    ]
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print(f"RESULT: all {len(results)} check groups PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main_entry()
