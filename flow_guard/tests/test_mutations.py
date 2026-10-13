"""The mutation table: every way a bad tool corrupts a flow, and what must happen."""

import copy

import pytest

import mutations as mut
from flow_guard import FlowIntegrityError, Policy, Severity, Verdict, assert_flow_preserved, compare


def run(flow, mutation, policy=None, judge=None):
    try:
        mutated = mutation(flow)
    except mut.NoTarget as exc:
        pytest.skip(f"fixture has no target: {exc}")
    before = copy.deepcopy(flow)
    report = compare(flow, mutated, policy=policy, judge=judge)
    assert flow == before, "compare() must never modify its inputs"
    return report


def severities(report):
    return {f.severity for f in report.findings}


def test_identity_is_pass_with_zero_findings(flow, fake_judge):
    report = run(flow, mut.identity, judge=fake_judge)
    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert report.topology_equivalent is True
    assert fake_judge.calls == []


def test_reverse_node_array_is_info_only(flow):
    report = run(flow, mut.reverse_nodes)
    assert report.verdict is Verdict.PASS
    assert report.categories() == {"node_reordered"}
    assert severities(report) == {Severity.INFO}


def test_consistent_id_rename_warns_and_topology_holds(flow):
    report = run(flow, mut.rename_ids)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    assert "node_id_changed" in report.categories()
    assert report.topology_equivalent is True
    assert Severity.CRITICAL not in severities(report)
    assert all(m.method == "content" for m in report.node_mapping)


def test_consistent_id_rename_fails_with_strict_ids(flow):
    report = run(flow, mut.rename_ids, policy=Policy(strict_ids=True))
    assert report.verdict is Verdict.FAIL
    assert {f.severity for f in report.by_category("node_id_changed")} == {Severity.CRITICAL}


def test_rename_with_one_dangling_edge_fails(flow):
    report = run(flow, mut.rename_ids_leave_dangling)
    assert report.verdict is Verdict.FAIL
    assert "dangling_reference" in report.categories()


def test_id_shift_is_detected(flow):
    report = run(flow, mut.shift_ids)
    assert report.verdict is Verdict.FAIL
    shift = report.by_category("node_id_shift")
    assert len(shift) == 1 and shift[0].severity is Severity.CRITICAL
    assert any(m.method == "id_shift" for m in report.node_mapping)
    # The content mapping is used: the redirect findings name the right content.
    assert report.topology_equivalent is False


def test_swapped_edge_destinations_fail(flow):
    report = run(flow, mut.swap_edge_destinations)
    assert report.verdict is Verdict.FAIL
    assert len(report.by_category("edge_redirected")) == 2


def test_deleted_edge_fails(flow):
    report = run(flow, mut.delete_edge)
    assert report.verdict is Verdict.FAIL
    assert "edge_removed" in report.categories()


def test_added_edge_fails(flow):
    report = run(flow, mut.add_edge)
    assert report.verdict is Verdict.FAIL
    assert "edge_added" in report.categories()


def test_one_word_condition_change_fails_with_char_diff(flow):
    report = run(flow, mut.change_condition_word)
    assert report.verdict is Verdict.FAIL
    (finding,) = report.by_category("condition_changed")
    assert "[-" in finding.unified_diff and "{+" in finding.unified_diff


def test_whitespace_only_condition_change_is_info(flow):
    report = run(flow, mut.whitespace_condition)
    assert report.verdict is Verdict.PASS
    assert report.categories() == {"condition_whitespace_only"}


def test_node_type_change_fails(flow):
    report = run(flow, mut.change_node_type)
    assert report.verdict is Verdict.FAIL
    assert "node_type_changed" in report.categories()


def test_start_node_change_fails(flow):
    report = run(flow, mut.change_start)
    assert report.verdict is Verdict.FAIL
    assert "start_node_changed" in report.categories()


def test_swapped_switch_cases_fail(flow):
    report = run(flow, mut.swap_switch_cases)
    assert report.verdict is Verdict.FAIL
    assert "switch_case_changed" in report.categories()


def test_removed_language_overlay_fails(flow):
    report = run(flow, mut.remove_overlay_language)
    assert report.verdict is Verdict.FAIL
    assert "overlay_language_removed" in report.categories()


def test_dropped_placeholder_fails_token_guard(flow):
    report = run(flow, mut.drop_placeholder)
    assert report.verdict is Verdict.FAIL
    (finding,) = report.by_category("token_lost")
    assert "{{devotee_name}}" in finding.message


@pytest.mark.parametrize("mutation", [mut.change_time_in_prompt, mut.change_amount_in_prompt],
                         ids=["time", "amount"])
def test_changed_time_or_amount_fails_token_guard(flow, mutation):
    report = run(flow, mutation)
    assert report.verdict is Verdict.FAIL
    assert "token_lost" in report.categories()


@pytest.mark.parametrize("mutation", [mut.corrupt_telugu_fffd, mut.corrupt_telugu_mojibake],
                         ids=["fffd", "mojibake"])
def test_corrupted_telugu_fails(flow, mutation):
    report = run(flow, mutation)
    assert report.verdict is Verdict.FAIL
    assert "text_corrupted" in report.categories()


def test_truncated_instruction_fails(flow):
    report = run(flow, mut.truncate_instruction)
    assert report.verdict is Verdict.FAIL
    assert "text_truncated" in report.categories()


def test_reworded_instruction_same_meaning_warns(flow, fake_judge):
    report = run(flow, mut.reword_instruction, judge=fake_judge)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    (finding,) = report.by_category("rewording_same_meaning")
    assert finding.source == "llm" and finding.severity is Severity.WARNING
    assert report.judge_summary


def test_reworded_instruction_without_judge_needs_review(flow):
    report = run(flow, mut.reword_instruction)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    (finding,) = report.by_category("text_needs_review")
    assert finding.unified_diff


def test_nudge_rule_change_is_meaning_changed(flow, fake_judge):
    report = run(flow, mut.nudge_meaning_change, judge=fake_judge)
    assert report.verdict is Verdict.FAIL
    (finding,) = report.by_category("meaning_changed")
    assert finding.source == "llm"
    assert "exactly one nudge" in finding.message


def test_reset_positions_is_info_only(flow):
    report = run(flow, mut.reset_positions)
    assert report.verdict is Verdict.PASS
    assert severities(report) == {Severity.INFO}
    assert "position_changed" in report.categories()


def test_nodes_list_to_dict_warns_only_on_container(flow):
    report = run(flow, mut.nodes_to_dict)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    warnings = [f for f in report.findings if f.severity is not Severity.INFO]
    assert [f.category for f in warnings] == ["nodes_container_changed"]


def test_reordered_edges_warn(flow):
    report = run(flow, mut.reorder_edges)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    assert "edge_order_changed" in report.categories()
    assert Severity.CRITICAL not in severities(report)


def test_unknown_key_warns_and_fails_when_strict(flow):
    report = run(flow, mut.add_unknown_key)
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    assert report.categories() == {"schema_key_added"}
    strict = run(flow, mut.add_unknown_key, policy=Policy(strict_schema=True))
    assert strict.verdict is Verdict.FAIL


def test_renamed_global_variable_fails(flow):
    report = run(flow, mut.rename_global_variable)
    assert report.verdict is Verdict.FAIL
    assert "global_variable_changed" in report.categories()


# --- judge can never downgrade deterministic failures ------------------------

class _LenientJudge:
    model = "lenient"

    def review_fields(self, node, fields, deterministic_findings):
        return [{"field_id": f["field_id"], "classification": "identical_meaning", "confidence": 1.0,
                 "evidence": [], "affected_rules": []} for f in fields]

    def summarize(self, digest):
        return ["All good."]


@pytest.mark.parametrize("mutation", [mut.drop_placeholder, mut.truncate_instruction,
                                      mut.corrupt_telugu_fffd, mut.change_condition_word])
def test_judge_cannot_downgrade_deterministic_failures(flow, mutation):
    report = run(flow, mutation, judge=_LenientJudge())
    assert report.verdict is Verdict.FAIL


def test_low_confidence_critical_becomes_needs_review(synthetic):
    from conftest import FakeJudge
    report = run(synthetic, mut.nudge_meaning_change, judge=FakeJudge(confidence=0.4))
    assert "meaning_changed" not in report.categories()
    (finding,) = report.by_category("needs_human_review")
    assert finding.severity is Severity.WARNING


def test_judge_failure_falls_back_to_needs_review(synthetic):
    class Broken:
        model = "broken"

        def review_fields(self, *a):
            raise RuntimeError("network down")

        def summarize(self, digest):
            raise RuntimeError("network down")

    report = run(synthetic, mut.reword_instruction, judge=Broken())
    assert report.verdict is Verdict.PASS_WITH_WARNINGS
    assert "text_needs_review" in report.categories()
    assert report.judge_status.startswith("partial")


def test_assert_flow_preserved(synthetic):
    assert assert_flow_preserved(synthetic, mut.reverse_nodes(synthetic)).verdict is Verdict.PASS
    with pytest.raises(FlowIntegrityError) as info:
        assert_flow_preserved(synthetic, mut.delete_edge(synthetic))
    assert info.value.report.verdict is Verdict.FAIL
    assert "edge_removed" in str(info.value)


# --- token guard counts occurrences, not just distinct tokens ------------------

def test_one_of_many_kb_amounts_changed_fails(flow):
    report = run(flow, mut.change_one_kb_amount)
    assert report.verdict is Verdict.FAIL
    (lost,) = report.by_category("token_lost")
    currency = next(t for t in lost.extra["tokens"] if t["kind"] == "currency")
    assert currency["token"] == "₹1008" and currency["before"] - currency["after"] == 1
    assert currency["after"] > 0  # the amount is still present elsewhere: a set compare would miss this
    assert "1 of" in lost.message and "Context in the input" in lost.message


def test_one_of_repeated_placeholders_deleted_fails(flow):
    report = run(flow, mut.delete_one_repeated_placeholder)
    assert report.verdict is Verdict.FAIL
    (lost,) = report.by_category("token_lost")
    (tok,) = [t for t in lost.extra["tokens"] if t["kind"] == "placeholder"]
    assert "devotee_name" in tok["token"] and tok["before"] >= 2 and tok["after"] == tok["before"] - 1


def test_shuffled_kb_rows_produce_no_token_findings(flow):
    report = run(flow, mut.shuffle_kb_rows)
    assert report.findings, "the mutation must actually change the prompt"
    assert not {"token_lost", "token_added"} & report.categories()
    assert Severity.CRITICAL not in severities(report)


def test_repeated_curly_placeholder_deleted_fails(synthetic):
    """{{devotee_name}} twice in one field, one removed (the real sample has no
    such field, so this case is built on the synthetic flow)."""
    src = copy.deepcopy(synthetic)
    node = src["data"]["conversationNodes"]["nodes"][1]
    node["onEnterSay"]["message"] = "Namaste {{devotee_name}} ji. Thank you, {{devotee_name}} ji, for your time."
    out = copy.deepcopy(src)
    out["data"]["conversationNodes"]["nodes"][1]["onEnterSay"]["message"] = (
        "Namaste {{devotee_name}} ji. Thank you, ji, for your time.")
    report = compare(src, out)
    assert report.verdict is Verdict.FAIL
    (lost,) = report.by_category("token_lost")
    assert "{{devotee_name}}: 1 of 2 lost (2 -> 1)" in lost.message


# --- false-alarm fixes: list markers and time formats (guard otherwise unchanged) --

def token_findings(report):
    return [f for f in report.findings if f.category in ("token_lost", "token_added")]


def test_numbered_list_to_bullets_has_no_token_findings(flow):
    report = run(flow, mut.kb_rows_to_bullets)
    assert report.findings, "the mutation must actually change the prompt"
    assert token_findings(report) == []
    assert Severity.CRITICAL not in severities(report)


def test_time_reformatted_has_no_token_findings(flow):
    report = run(flow, lambda d: mut.rewrite_time(d, "10 AM", "10:00 AM"))
    assert report.findings, "the mutation must actually change the prompt"
    assert token_findings(report) == []
    assert Severity.CRITICAL not in severities(report)


@pytest.mark.parametrize("new", ["10 PM", "10:30 AM"])
def test_time_value_changed_is_critical(flow, new):
    report = run(flow, lambda d: mut.rewrite_time(d, "10 AM", new))
    assert report.verdict is Verdict.FAIL
    (lost,) = report.by_category("token_lost")
    assert lost.severity is Severity.CRITICAL
    assert [(t["kind"], t["token"]) for t in lost.extra["tokens"]] == [("time", "10 AM")]


def test_one_kb_amount_changed_still_critical_after_marker_fix(flow):
    report = run(flow, mut.change_one_kb_amount)
    (lost,) = report.by_category("token_lost")
    assert lost.severity is Severity.CRITICAL
    currency = next(t for t in lost.extra["tokens"] if t["kind"] == "currency")
    assert (currency["before"], currency["after"]) == (9, 8)


def test_inline_number_changed_is_critical(flow):
    report = run(flow, mut.change_inline_number)
    assert report.verdict is Verdict.FAIL
    (lost,) = report.by_category("token_lost")
    assert any(t["kind"] == "number" for t in lost.extra["tokens"])
