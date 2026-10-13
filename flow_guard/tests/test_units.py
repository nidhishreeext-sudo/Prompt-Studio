"""Loader, structure, tokens and policy."""

import copy
import json

import pytest

import mutations as mut
from flow_guard import FlowLoadError, Policy, PolicyError, Severity, compare, load_flow, load_policy, validate
from flow_guard.diff import char_diff
from flow_guard.tokens import corruption_reasons, extract_tokens, normalize_text


# --- loader --------------------------------------------------------------------

def test_sample_shape(flow):
    loaded = load_flow(flow)
    assert loaded.envelope == "envelope"
    assert {n["type"] for n in loaded.nodes} <= {"conversation", "switch", "end"}


def test_synthetic_has_7_nodes_and_11_edges(synthetic):
    loaded = load_flow(synthetic)
    assert len(loaded.nodes) == 7
    assert sum(len(n["edges"]) for n in loaded.nodes) == 11


def test_bare_data_and_bare_conversation_nodes_are_accepted(synthetic):
    data = synthetic["data"]
    assert load_flow(data).envelope == "data"
    assert load_flow(data["conversationNodes"]).envelope == "conversation_nodes"
    # Same content in a different wrapper: only the wrapper finding.
    report = compare(synthetic, data)
    assert report.categories() == {"envelope_changed"}


def test_loader_does_not_mutate_dict_input(synthetic):
    as_dict = mut.nodes_to_dict(synthetic)
    for node in as_dict["data"]["conversationNodes"]["nodes"].values():
        del node["id"]
    before = copy.deepcopy(as_dict)
    loaded = load_flow(as_dict)
    assert [n["id"] for n in loaded.nodes] == [n["id"] for n in synthetic["data"]["conversationNodes"]["nodes"]]
    assert as_dict == before


@pytest.mark.parametrize("bad", ["{not json", "[]", '{"data": {"prompt": "x"}}', '{"data": {"conversationNodes": {"nodes": 5}}}'])
def test_loader_rejects_non_flows(bad):
    with pytest.raises(FlowLoadError):
        load_flow(bad)


def test_loader_reports_missing_file(tmp_path):
    with pytest.raises(FlowLoadError):
        load_flow(tmp_path / "nope.json")


# --- structure -------------------------------------------------------------------

def test_valid_flow_has_no_structural_findings(flow):
    assert validate(flow) == []


def _broken(doc, fn):
    doc = copy.deepcopy(doc)
    fn(doc["data"]["conversationNodes"])
    return validate(doc)


def _node(conv, node_type):
    return next(n for n in conv["nodes"] if n["type"] == node_type)


@pytest.mark.parametrize("category,breaker", [
    ("start_node_missing", lambda c: c.update(startNodeId="ghost")),
    ("dangling_reference", lambda c: _node(c, "conversation")["edges"][0].update(destinationNodeId="ghost")),
    ("duplicate_node_id", lambda c: c["nodes"][2].update(id=c["nodes"][1]["id"])),
    ("duplicate_edge_id", lambda c: _node(c, "conversation")["edges"][1].update(id=_node(c, "conversation")["edges"][0]["id"])),
    ("switch_default_invalid", lambda c: _node(c, "switch").update(switchDefaultEdgeId="ghost-edge")),
    ("switch_case_invalid", lambda c: _node(c, "switch")["switchCases"].update(NEW_CASE="ghost-edge")),
    ("switch_expression_missing", lambda c: _node(c, "switch").update(switchExpression="")),
    ("end_node_has_edges", lambda c: _node(c, "end")["edges"].append(copy.deepcopy(_node(c, "conversation")["edges"][0]) | {"id": "x-new"})),
    ("dead_end_node", lambda c: _node(c, "conversation").update(edges=[])),
    ("duplicate_global_variable", lambda c: c["globalVariables"].append(dict(c["globalVariables"][0]))),
    ("empty_node_id", lambda c: c["nodes"][-1].update(id="")),
])
def test_structure_detects(synthetic, category, breaker):
    findings = _broken(synthetic, breaker)
    assert category in {f.category for f in findings}, [f.category for f in findings]


def test_unreachable_and_no_path_to_end(synthetic):
    def cut(conv):
        for node in conv["nodes"]:
            for edge in node["edges"]:
                if conv["nodes"][-1]["id"] == edge["destinationNodeId"]:
                    edge["destinationNodeId"] = conv["nodes"][-2]["id"]
    assert "unreachable_node" in {f.category for f in _broken(synthetic, cut)}

    def no_end(conv):
        for node in conv["nodes"]:
            if node["type"] == "end":
                node["type"] = "conversation"
                node["edges"] = [{"id": f"loop-{node['id']}", "destinationNodeId": conv["startNodeId"],
                                  "transitionCondition": {"type": "prompt", "prompt": "again"}}]
    assert "no_path_to_end" in {f.category for f in _broken(synthetic, no_end)}


def test_empty_switch_conditions_are_not_flagged(synthetic):
    switch = _node(synthetic["data"]["conversationNodes"], "switch")
    assert all(e["transitionCondition"]["prompt"] == "" for e in switch["edges"])
    assert validate(synthetic) == []


def test_preexisting_input_problems_do_not_fail_compare(synthetic):
    broken = copy.deepcopy(synthetic)
    _node(broken["data"]["conversationNodes"], "end")["edges"].append(
        {"id": "weird", "destinationNodeId": broken["data"]["conversationNodes"]["startNodeId"],
         "transitionCondition": {"type": "prompt", "prompt": "restart"}})
    report = compare(broken, copy.deepcopy(broken))
    assert report.findings == []
    assert [f.category for f in report.input_issues] == ["end_node_has_edges"]


# --- tokens --------------------------------------------------------------------

def test_token_extraction():
    tokens = extract_tokens(r"Call \[\[devotee_name]] at 10 AM or 7:30 pm, pay ₹1,100 or Rs. 501. "
                            "See https://x.org/pay/{{booking_id}}. Total 3 items.")
    assert tokens["placeholder"] == {"[[devotee_name]]", "{{booking_id}}"}
    assert tokens["time"] == {"10 AM", "7:30 PM"}
    assert tokens["currency"] == {"₹1100", "₹501"}
    assert tokens["url"] == {"https://x.org/pay/{{booking_id}}"}
    assert {"1100", "501", "3"} <= tokens["number"]
    assert "10" not in tokens["number"]  # the 10 belongs to the time "10 AM"


def test_escaped_and_plain_markers_are_the_same_token():
    assert extract_tokens(r"\[\[x]]")["placeholder"] == extract_tokens("[[x]]")["placeholder"]
    assert normalize_text(r"hi \[\[x]]") == normalize_text("hi [[x]]")


def test_native_digits_are_numbers():
    assert "10" in extract_tokens("सुबह १० बजे")["number"]


def test_nfc_only_change_is_info(synthetic):
    import unicodedata
    out = copy.deepcopy(synthetic)
    node = out["data"]["conversationNodes"]["nodes"][1]
    node["instruction"] = unicodedata.normalize("NFD", "Café " + node["instruction"])
    src = copy.deepcopy(synthetic)
    src["data"]["conversationNodes"]["nodes"][1]["instruction"] = "Café " + node["instruction"][5:]
    src["data"]["conversationNodes"]["nodes"][1]["instruction"] = unicodedata.normalize(
        "NFC", src["data"]["conversationNodes"]["nodes"][1]["instruction"])
    report = compare(src, out)
    assert report.categories() == {"text_normalization_only"}
    assert {f.severity for f in report.findings} == {Severity.INFO}


def test_corruption_detection_ignores_clean_text():
    text = "నమస్తే గారు, ఎలా ఉన్నారు? Café à la carte."
    assert corruption_reasons(text, text + " ok") == []
    assert corruption_reasons(text, text.replace("న", "�", 1))


def test_char_diff_reads_as_word_edit():
    assert char_diff("The devotee agrees.", "The devotee refuses.") == "The devotee [-agrees-]{+refuses+}."


# --- policy -------------------------------------------------------------------

def test_policy_override_file(tmp_path, synthetic):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"severities": {"edge_order_changed": "CRITICAL"}, "thresholds": {"match_threshold": 0.7}}))
    policy = load_policy(path)
    assert policy.severity("edge_order_changed") is Severity.CRITICAL
    report = compare(synthetic, mut.reorder_edges(synthetic), policy=policy)
    assert report.verdict.value == "FAIL"


@pytest.mark.parametrize("raw", [{"severities": {"no_such_check": "INFO"}}, {"severities": {"edge_added": "LOUD"}},
                                 {"thresholds": {"match_threshold": 3}}, {"bogus": 1}])
def test_policy_rejects_bad_overrides(raw):
    with pytest.raises(PolicyError):
        Policy.from_dict(raw)


def test_every_category_used_has_a_default():
    import re
    from pathlib import Path
    from flow_guard.policy import DEFAULT_SEVERITIES
    src = Path(__file__).resolve().parent.parent / "src" / "flow_guard"
    used = set()
    for py in src.glob("*.py"):
        used |= set(re.findall(r'(?:emit|severity|_f)\(\s*(?:policy,\s*)?"([a-z_]+)"', py.read_text(encoding="utf-8")))
    assert used and used <= set(DEFAULT_SEVERITIES), used - set(DEFAULT_SEVERITIES)


def test_token_delta_counts_occurrences():
    from flow_guard.tokens import token_delta
    before = "Pay ₹1,008 or ₹5,008. Most give ₹1,008. Call {{name}}, yes {{name}}."
    lost, added = token_delta(before, before.replace("Most give ₹1,008", "Most give ₹999"))
    assert [(t.kind, t.token, t.before, t.after) for t in lost] == [
        ("number", "1008", 2, 1), ("currency", "₹1008", 2, 1)]
    assert {(t.kind, t.token) for t in added} == {("number", "999"), ("currency", "₹999")}
    lost, added = token_delta(before, before.replace(", yes {{name}}", ""))
    assert [(t.token, t.before, t.after) for t in lost] == [("{{name}}", 2, 1)]
    assert added == []


def test_token_delta_ignores_reordering():
    from flow_guard.tokens import token_delta
    before = "Slots: 10 AM, 7 PM. Amounts ₹501 / ₹1,008 / ₹501. See https://a.org and {{x}} {{y}}."
    after = "Amounts ₹501 / ₹501 / ₹1,008. {{y}} {{x}}, see https://a.org. Slots: 7 PM, 10 AM."
    assert token_delta(before, after) == ([], [])


def test_list_markers_are_not_numbers_but_inline_numbers_are():
    from flow_guard.tokens import token_counts
    text = "1. Offer 3 sevas.\n  2) Wait 10 seconds.\n10. Done in 2. steps"
    numbers = token_counts(text)["number"]
    assert numbers == {"3": 1, "10": 1, "2": 1}


def test_time_formats_normalize():
    from flow_guard.tokens import token_counts
    same = ["10 AM", "10:00 AM", "10am", "10 a.m.", "10:00am"]
    assert {tuple(token_counts(t)["time"]) for t in same} == {("10 AM",)}
    distinct = {next(iter(token_counts(t)["time"])) for t in ["10 AM", "10:30 AM", "10 PM", "7 AM"]}
    assert len(distinct) == 4
    assert not token_counts("10:00 AM")["number"]  # minutes are not loose numbers
