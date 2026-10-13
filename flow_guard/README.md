# flow_guard

Integrity checker and diff tool for **guided conversation flow** JSON (voice-agent flows with nodes, edges, transition conditions, prompts and per-language overlays).

Give it the flow you started with (INPUT) and the flow a tool produced from it (OUTPUT). It answers two questions:

1. **Structure:** are the same nodes, connections and transition conditions still there?
2. **Diff:** what exactly changed, and is each change acceptable?

Deterministic rules decide everything they can. An optional OpenAI judge classifies only the text changes that rules can't decide, such as whether a rewritten instruction still means the same thing. The judge can add findings but never removes or downgrades a rule-based failure.

Standalone: Python 3.10+, standard library plus `openai`. It never modifies its input files.

## Install

```bash
pip install -e ./flow_guard            # from the folder that contains flow_guard/
pip install -e "./flow_guard[dev]"     # adds pytest
```

## Command line

```bash
flow-guard check INPUT OUTPUT [--policy policy.json] [--no-llm] [--model M]
                              [--report-dir DIR] [--strict-ids] [--strict-schema]
                              [--fail-on-warning]
flow-guard validate FLOW
```

Examples:

```bash
# Rules only, no API calls
flow-guard check samples/input_flow.json exported.json --no-llm

# With the AI judge (needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
flow-guard check samples/input_flow.json exported.json --report-dir reports/run1

# Treat any id change or unknown key as a failure, and fail CI on warnings
flow-guard check in.json out.json --strict-ids --strict-schema --fail-on-warning

# Check one flow on its own
flow-guard validate exported.json
```

`python -m flow_guard ...` works too.

| Exit code | Meaning |
| --- | --- |
| 0 | PASS |
| 1 | PASS_WITH_WARNINGS (2 with `--fail-on-warning`) |
| 2 | FAIL |
| 3 | Usage error, unreadable file, invalid JSON, not a flow, bad policy file |

`check` writes `report.json` (machine-readable) and `report.md` (for people) to `--report-dir` (default `flow_guard_report/`).

**Verdict:** FAIL if any finding is CRITICAL, PASS_WITH_WARNINGS if any is WARNING, otherwise PASS.

## Python API

```python
from flow_guard import compare, validate, assert_flow_preserved, FlowIntegrityError
from flow_guard import OpenAIJudge, Policy, load_policy
from flow_guard.report import write_reports

# Flows can be paths, JSON strings, or dicts. Inputs are never modified.
report = compare("before.json", "after.json")                       # rules only
report = compare(before_dict, after_dict, judge=OpenAIJudge())      # with the judge

report.verdict            # Verdict.PASS / PASS_WITH_WARNINGS / FAIL
report.counts             # {"CRITICAL": 0, "WARNING": 2, "INFO": 1}
report.topology_equivalent
for f in report.findings:
    print(f.severity.value, f.category, f.message)
write_reports(report, "reports/")                                   # report.json + report.md

findings = validate("flow.json")                                     # one flow, structure only

# Pipeline guard: raises FlowIntegrityError (with .report) on FAIL
try:
    assert_flow_preserved(before_dict, after_dict, policy=Policy(strict_ids=True))
except FlowIntegrityError as err:
    print(err)               # lists the critical findings
    err.report               # the full Report
```

Accepted flow shapes: the full export `{version, type, exportedAt, data: {...}}`, the bare `data` object, or the bare `conversationNodes` object. `nodes` may be a list or a dict keyed by node id.

## What gets checked

**Single flow (`validate`, and both sides of `check`):** required keys, unique non-empty node ids, edge ids unique across the flow, start node exists, every edge, goBackEdge and jumpToEdge destination exists, switch cases and default point at the switch's own edges, switch nodes have an expression, end nodes have no outgoing edges, other nodes have at least one, every node is reachable, some path reaches an end node, global variable names are unique. In `check`, problems the input already had are listed separately under "Problems already present in the original" and don't count toward the verdict.

**Input vs output (`check`):**

| Area | Rule (default severity) |
| --- | --- |
| Node matching | Match by id, then by content score (edge conditions 40%, instruction 25%, name 15%, onEnterSay 10%, type 10%). Ids shifted onto other nodes' content gives `node_id_shift` (CRITICAL), and the content mapping is used from then on. Same content under a new id gives `node_id_changed` (WARNING; CRITICAL with `--strict-ids`). Removed or added nodes are CRITICAL. |
| Nodes | Type change CRITICAL, name change WARNING, start node change CRITICAL. |
| Edges | Removed, added or redirected (after mapping): CRITICAL. Condition text changed: CRITICAL, with a character-level diff. Whitespace-only condition change: INFO. Edge id change: WARNING. Edge order change: WARNING. `goBackToPrevious` change: CRITICAL. Switch edges with empty conditions are normal and are not flagged. |
| Graph | Compares transitions (source, destination, condition hash), the nodes reachable from start, and the end nodes. Not equivalent: CRITICAL. `report.topology_equivalent` always holds the result. |
| Switch | Expression, each case's destination and the default destination must match (CRITICAL). |
| Protected | globalVariables (name, type), supportedModes, allowGoBack, goBackEdges, jumpToEdges, globalNode.enabled: any change is CRITICAL. |
| Token guard | In every text field: `{{var}}` and `[[var]]` placeholders (`\[\[x]]` is the same as `[[x]]`), numbers (including Devanagari or Telugu digits; the `1.` / `2)` marker at the start of a markdown list line is not counted), ₹ and Rs amounts, times ("10 AM", "10:00 AM", "10am" and "10 a.m." are the same time; "10:30 AM" and "10 PM" are not), and URLs. Tokens are counted, not just collected: if ₹1,008 appears 9 times and the output has 8, one is lost. Any count that goes down is CRITICAL (the finding says how many and quotes the surrounding text); any count that goes up is WARNING; the same tokens in a different order produce nothing. Runs with or without the judge. |
| Translations | Losing a `lang:*` overlay on a node or edge, or a `data.prompts` key, is CRITICAL. Adding one is WARNING. |
| Text | Compared after NFC normalization. Whitespace-only or normalization-only: INFO. Truncated (output is a prefix of input): CRITICAL. Corrupted (U+FFFD, mojibake, a script that vanished): CRITICAL. Anything else goes to the judge, or becomes a WARNING "needs review" with a diff when the judge is off. |
| Cosmetic | Positions, startNodePosition, exportedAt, node array order and key order are INFO. Nodes list turned into a dict (or the reverse) is WARNING. |
| Schema | Added or removed keys and type changes (e.g. number to string) are WARNING (CRITICAL with `--strict-schema`). Other changed settings (tts, stt, callSettings...) are WARNING. |

## Keys

The judge reads `OPENAI_API_KEY` from the environment. Set it in one of two ways:

- As an environment variable for the session or user (e.g. through the Windows "Environment Variables" dialog, or your CI's secret store).
- In a `.env` file that git ignores (`flow_guard/.gitignore` already ignores `.env` and `.env.*`) and that your shell or runner loads. flow_guard does not read `.env` files itself.

Never paste the key into a command line. Commands are saved to shell history (PowerShell keeps `%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt`) and often end up in shared terminals, screenshots or chat. flow_guard never writes the key to `.flow_guard_cache/`, `report.json` or `report.md`; a test checks this.

## The AI judge

| Env var | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required to enable the judge. Never logged or written to the cache. |
| `OPENAI_JUDGE_MODEL` | Model (default `gpt-6-luna`; `--model` overrides). |
| `OPENAI_BASE_URL` | Optional, for proxies or compatible endpoints. |
| `FLOW_GUARD_CACHE_DIR` | Response cache (default `.flow_guard_cache/` in the working directory). |

- One call per node that has rewritten text (all its fields together), plus one call for a plain-English summary of at most 10 bullets.
- Uses JSON-schema structured output and temperature 0. If the model rejects `temperature`, the call is retried without it. Timeouts and transient errors are retried with backoff.
- The system prompt treats flow text as data and ignores any instructions inside it. It also tells the model that rule-based findings are ground truth and that any change to a number, limit, condition, prohibition or sequence counts as `meaning_changed`.
- Severity of each classification: `meaning_changed`, `content_removed` and `corrupted` are CRITICAL. `content_added` and `rewording_same_meaning` are WARNING. `identical_meaning` is INFO. A CRITICAL verdict with confidence under 0.6 becomes WARNING `needs_human_review`.
- Responses are cached by `sha256(model + prompt version + payload)`, so reruns are free and repeatable.
- With no key or with `--no-llm`, the judge is skipped and the report says so. If a call fails, the affected fields fall back to "needs review" and the check still completes.

To use a different judge, pass any object with a `model` attribute and the methods `review_fields(node, fields, deterministic_findings)` and `summarize(digest)`. See `flow_guard/judge.py` for the contract.

## Policy override

All severities and thresholds live in `src/flow_guard/policy.py`. Override them with a JSON file:

```json
{
  "severities": {
    "node_name_changed": "INFO",
    "edge_order_changed": "CRITICAL",
    "token_added": "INFO"
  },
  "thresholds": {
    "match_threshold": 0.6,
    "id_shift_low": 0.5,
    "id_shift_high": 0.8,
    "judge_min_confidence": 0.6
  },
  "strict_ids": false,
  "strict_schema": false
}
```

Unknown categories, thresholds or severities are rejected (exit code 3), so a typo can't silently do nothing. The category names are the keys of `DEFAULT_SEVERITIES` in `policy.py` and match the `category` field in reports.

## Embedding in another project

The package has no imports from any host project. Either:

- **Install it:** add `-e ./flow_guard` (or a path or git URL) to the host's requirements, or
- **Vendor it:** copy `flow_guard/src/flow_guard/` into the host and make sure `openai` is installed.

A typical guard after an import/export step:

```python
from flow_guard import assert_flow_preserved, FlowIntegrityError, OpenAIJudge, judge_from_env

judge, status = judge_from_env()   # None when OPENAI_API_KEY is missing
try:
    report = assert_flow_preserved(original_flow, exported_flow, judge=judge)
    if report.verdict.value == "PASS_WITH_WARNINGS":
        log.warning("flow changed: %s", [f.message for f in report.findings if f.severity.value == "WARNING"])
except FlowIntegrityError as err:
    raise RuntimeError(f"export corrupted the flow:\n{err}") from err
```

## Development

```bash
cd flow_guard
pytest                 # offline; the judge is mocked
pytest -m live         # one real judge call; skipped without OPENAI_API_KEY
```

`tests/mutations.py` holds helpers that corrupt a flow the way a bad tool would. Every mutation runs against `tests/fixtures/synthetic_flow.json` (7 nodes, 11 edges, switch routing, 5-language overlays) and, when present, against `samples/input_flow.json`.
