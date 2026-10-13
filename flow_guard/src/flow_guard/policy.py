"""Severity policy. Every severity a check can emit lives here, never in the checks.

Override with a JSON file:

    {
      "severities": {"node_name_changed": "INFO", "edge_order_changed": "CRITICAL"},
      "thresholds": {"match_threshold": 0.7},
      "strict_ids": true,
      "strict_schema": false
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import Severity

C, W, I = Severity.CRITICAL, Severity.WARNING, Severity.INFO

DEFAULT_SEVERITIES: dict[str, Severity] = {
    # --- A. single-flow structure -------------------------------------------
    "missing_key": C,
    "invalid_value": C,
    "empty_node_id": C,
    "duplicate_node_id": C,
    "duplicate_edge_id": C,
    "start_node_missing": C,
    "dangling_reference": C,
    "switch_case_invalid": C,
    "switch_default_invalid": C,
    "switch_expression_missing": C,
    "end_node_has_edges": C,
    "dead_end_node": C,
    "unreachable_node": C,
    "no_path_to_end": C,
    "duplicate_global_variable": C,
    # --- B1. node matching ---------------------------------------------------
    "node_id_shift": C,
    "node_id_changed": W,
    "node_removed": C,
    "node_added": C,
    # --- B2/B3. node properties, start node ---------------------------------
    "node_type_changed": C,
    "node_name_changed": W,
    "start_node_changed": C,
    # --- B4. edges -----------------------------------------------------------
    "edge_removed": C,
    "edge_added": C,
    "edge_redirected": C,
    "condition_changed": C,
    "condition_emptied": C,
    "condition_whitespace_only": I,
    "edge_id_changed": W,
    "edge_order_changed": W,
    "go_back_to_previous_changed": C,
    # --- B5. graph equivalence ----------------------------------------------
    "topology_equivalent": I,
    "topology_not_equivalent": C,
    # --- B6. switch nodes ----------------------------------------------------
    "switch_expression_changed": C,
    "switch_case_changed": C,
    "switch_default_changed": C,
    # --- B7. protected fields -----------------------------------------------
    "global_variable_changed": C,
    "supported_modes_changed": C,
    "allow_go_back_changed": C,
    "go_back_edges_changed": C,
    "jump_to_edges_changed": C,
    "global_node_changed": C,
    # --- B8. token guard ------------------------------------------------------
    "token_lost": C,
    "token_added": W,
    # --- B9. overlays / translations ----------------------------------------
    "overlay_language_removed": C,
    "overlay_language_added": W,
    "prompt_key_removed": C,
    "prompt_key_added": W,
    # --- B10. text classification -------------------------------------------
    "text_whitespace_only": I,
    "text_normalization_only": I,
    "text_truncated": C,
    "text_corrupted": C,
    "text_needs_review": W,
    # --- LLM judge ------------------------------------------------------------
    "meaning_changed": C,
    "content_removed": C,
    "content_added": W,
    "corrupted": C,
    "rewording_same_meaning": W,
    "identical_meaning": I,
    "needs_human_review": W,
    # --- B11. cosmetic --------------------------------------------------------
    "position_changed": I,
    "exported_at_changed": I,
    "node_reordered": I,
    "key_order_changed": I,
    "nodes_container_changed": W,
    "envelope_changed": W,
    # --- B12. schema drift / everything else --------------------------------
    "schema_key_added": W,
    "schema_key_removed": W,
    "type_drift": W,
    "field_changed": W,
}

# Categories whose severity is raised when a strict flag is on.
STRICT_IDS_OVERRIDES = {"node_id_changed": C, "edge_id_changed": C}
STRICT_SCHEMA_OVERRIDES = {"schema_key_added": C, "schema_key_removed": C, "type_drift": C}

DEFAULT_THRESHOLDS: dict[str, float] = {
    # node matching (B1)
    "id_shift_low": 0.5,     # id-matched pair below this is suspicious ...
    "id_shift_high": 0.8,    # ... if another output node scores at least this
    "match_threshold": 0.6,  # pass-2 content match
    # judge
    "judge_min_confidence": 0.6,
}

SCORE_WEIGHTS: dict[str, float] = {
    "edges": 0.40,
    "instruction": 0.25,
    "name": 0.15,
    "on_enter_say": 0.10,
    "type": 0.10,
}


class PolicyError(ValueError):
    pass


@dataclass
class Policy:
    severities: dict[str, Severity] = field(default_factory=lambda: dict(DEFAULT_SEVERITIES))
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    strict_ids: bool = False
    strict_schema: bool = False

    def severity(self, category: str) -> Severity:
        if self.strict_ids and category in STRICT_IDS_OVERRIDES:
            return STRICT_IDS_OVERRIDES[category]
        if self.strict_schema and category in STRICT_SCHEMA_OVERRIDES:
            return STRICT_SCHEMA_OVERRIDES[category]
        try:
            return self.severities[category]
        except KeyError:
            raise PolicyError(f"no severity defined for category {category!r}") from None

    def threshold(self, name: str) -> float:
        return self.thresholds[name]

    def with_flags(self, strict_ids: bool | None = None, strict_schema: bool | None = None) -> "Policy":
        return Policy(
            severities=dict(self.severities),
            thresholds=dict(self.thresholds),
            strict_ids=self.strict_ids if strict_ids is None else strict_ids,
            strict_schema=self.strict_schema if strict_schema is None else strict_schema,
        )

    @classmethod
    def from_dict(cls, raw: dict) -> "Policy":
        if not isinstance(raw, dict):
            raise PolicyError("policy file must contain a JSON object")
        unknown_top = set(raw) - {"severities", "thresholds", "strict_ids", "strict_schema"}
        if unknown_top:
            raise PolicyError(f"unknown policy keys: {sorted(unknown_top)}")
        policy = cls()
        for category, value in (raw.get("severities") or {}).items():
            if category not in DEFAULT_SEVERITIES:
                raise PolicyError(f"unknown finding category in policy: {category!r}")
            try:
                policy.severities[category] = Severity.parse(value)
            except ValueError as exc:
                raise PolicyError(str(exc)) from None
        for name, value in (raw.get("thresholds") or {}).items():
            if name not in DEFAULT_THRESHOLDS:
                raise PolicyError(f"unknown threshold in policy: {name!r}")
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise PolicyError(f"threshold {name!r} must be a number between 0 and 1")
            policy.thresholds[name] = float(value)
        policy.strict_ids = bool(raw.get("strict_ids", False))
        policy.strict_schema = bool(raw.get("strict_schema", False))
        return policy


def load_policy(path: str | Path) -> Policy:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"cannot read policy file {path}: {exc}") from exc
    return Policy.from_dict(raw)
