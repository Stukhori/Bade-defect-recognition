"""Frozen review-field schemas loaded from the Phase 9A configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from windblade.config import load_config


class ReviewSchemaError(ValueError):
    """Raised when the frozen review schema is absent or malformed."""


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    choices: tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class PassSchema:
    version: str
    pass_name: str
    fields: tuple[FieldDefinition, ...]

    @property
    def headers(self) -> tuple[str, ...]:
        return ("review_id", *(field.name for field in self.fields))

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.required)

    @property
    def notes_field(self) -> str:
        notes = [field.name for field in self.fields if not field.required]
        if notes != ["reviewer_notes"]:
            raise ReviewSchemaError("the frozen schema must contain one optional reviewer_notes field")
        return notes[0]

    def definition(self, name: str) -> FieldDefinition:
        for field in self.fields:
            if field.name == name:
                return field
        raise ReviewSchemaError(f"unknown review field: {name}")


def _choice_text(value: Any) -> str:
    # PyYAML's YAML 1.1 resolver reads the frozen unquoted yes/no tokens as
    # booleans. Convert them back to their exact intended CSV spellings.
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, str) and value:
        return value
    raise ReviewSchemaError(f"invalid frozen response choice: {value!r}")


def load_pass_schema(config_path: str | Path, pass_name: str) -> PassSchema:
    """Load exactly one pass schema so Pass B need not be loaded early."""

    if pass_name not in {"pass_a", "pass_b"}:
        raise ReviewSchemaError(f"unsupported review pass: {pass_name}")
    data = load_config(Path(config_path)).as_dict()
    packet = data.get("review_packet")
    if not isinstance(packet, dict) or packet.get("version") != "phase9a_blinded_two_pass_v1":
        raise ReviewSchemaError("unexpected or missing Phase 9A review schema version")
    raw_fields = packet.get(f"{pass_name}_fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ReviewSchemaError(f"missing frozen fields for {pass_name}")
    fields: list[FieldDefinition] = []
    for name, raw_choices in raw_fields.items():
        if name == "reviewer_notes":
            if raw_choices != "free_text":
                raise ReviewSchemaError("reviewer_notes must remain optional free text")
            fields.append(FieldDefinition(name=name, choices=(), required=False))
            continue
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ReviewSchemaError(f"required field has no choices: {name}")
        choices = tuple(_choice_text(choice) for choice in raw_choices)
        if len(set(choices)) != len(choices):
            raise ReviewSchemaError(f"duplicate choices in frozen field: {name}")
        fields.append(FieldDefinition(name=name, choices=choices, required=True))
    schema = PassSchema(str(packet["version"]), pass_name, tuple(fields))
    _validate_expected_identity(schema)
    return schema


def _validate_expected_identity(schema: PassSchema) -> None:
    expected = {
        "pass_a": (
            "review_id",
            "defect_visible",
            "corruption_obscures_diagnostic_detail",
            "dataset_label_visually_plausible",
            "visually_ambiguous_between_categories",
            "possible_crop_or_background_problem",
            "reviewer_notes",
        ),
        "pass_b": (
            "review_id",
            "activation_primarily_inside_annotation",
            "activation_concentrated_on_degradation_artifact",
            "pattern_consistent_across_cnn_seeds",
            "prediction_visually_understandable_after_reveal",
            "reviewer_notes",
        ),
    }
    if schema.headers != expected[schema.pass_name]:
        raise ReviewSchemaError(f"frozen {schema.pass_name} field order changed")
