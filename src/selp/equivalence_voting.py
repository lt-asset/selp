"""Equivalence voting for NL-to-LTL model outputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .io import read_jsonl, read_records, write_jsonl
from .ltl import are_equivalent, is_valid_formula

EquivalenceFn = Callable[[str, str], bool]
ValidationFn = Callable[[str], bool]


@dataclass(frozen=True)
class FormulaGroup:
    formula: str
    members: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.members)

    def as_vote_item(self) -> list[Any]:
        return [self.formula, {"count": self.count, "syntax": False}]


def group_equivalent_formulas(
    candidates: Sequence[str],
    equivalent: EquivalenceFn = are_equivalent,
    validate: ValidationFn = is_valid_formula,
) -> tuple[list[FormulaGroup], list[str]]:
    """Group syntactically valid formulas by semantic equivalence."""

    groups: list[FormulaGroup] = []
    syntax_errors: list[str] = []

    for formula in candidates:
        if not validate(formula):
            syntax_errors.append(formula)
            continue

        for index, group in enumerate(groups):
            if equivalent(formula, group.formula):
                groups[index] = FormulaGroup(group.formula, group.members + (formula,))
                break
        else:
            groups.append(FormulaGroup(formula, (formula,)))

    groups.sort(key=lambda group: group.count, reverse=True)
    return groups, syntax_errors


def majority_vote(
    candidates: Sequence[str],
    ground_truth_formula: str | None = None,
    equivalent: EquivalenceFn = are_equivalent,
    validate: ValidationFn = is_valid_formula,
) -> dict[str, Any]:
    """Return the majority-equivalence result for a set of generated formulas."""

    groups, syntax_errors = group_equivalent_formulas(candidates, equivalent, validate)
    first_valid = next((formula for formula in candidates if validate(formula)), None)
    majority_formula = groups[0].formula if groups else "SyntaxError"

    result: dict[str, Any] = {
        "majority_formula": majority_formula,
        "vote_list": [group.as_vote_item() for group in groups],
        "syntax_errors": syntax_errors,
        "only_one_formula": first_valid,
        "only_one_correct": None,
        "majority_vote_acc": None,
    }

    if ground_truth_formula is None:
        return result

    if first_valid is not None:
        result["only_one_correct"] = equivalent(first_valid, ground_truth_formula)
    else:
        result["only_one_correct"] = False

    if groups:
        result["majority_vote_acc"] = equivalent(majority_formula, ground_truth_formula)
    else:
        result["majority_vote_acc"] = False

    return result


def vote_record(
    record: dict[str, Any],
    ground_truth_formula: str | None = None,
    candidates_field: str | None = None,
    equivalent: EquivalenceFn = are_equivalent,
    validate: ValidationFn = is_valid_formula,
) -> dict[str, Any]:
    """Apply equivalence voting to one prediction record."""

    field = candidates_field or _find_candidates_field(record)
    candidates = record.get(field)
    if candidates is None:
        raise KeyError(f"Could not find candidate formula field in record: {record.keys()}")
    if isinstance(candidates, str):
        candidates = [candidates]

    if ground_truth_formula is None:
        ground_truth_formula = record.get("formula") or record.get("ground_truth_formula")

    voted = majority_vote(
        list(candidates),
        ground_truth_formula=ground_truth_formula,
        equivalent=equivalent,
        validate=validate,
    )
    output = dict(record)
    output.update(voted)
    if ground_truth_formula is not None:
        output["ground_truth_formula"] = ground_truth_formula
    return output


def run_equivalence_voting(
    input_path: str,
    output_path: str,
    ground_truth_path: str | None = None,
    id_field: str = "task_ID",
    formula_field: str = "formula",
    candidates_field: str | None = None,
) -> None:
    """Vote over an input JSONL file and write JSONL results."""

    records = read_jsonl(input_path)
    ground_truth = _load_ground_truth(ground_truth_path, id_field, formula_field)

    output_records = []
    for index, record in enumerate(records):
        record_id = record.get(id_field, index)
        formula = ground_truth.get(record_id) or ground_truth.get(str(record_id))
        output_records.append(
            vote_record(record, formula, candidates_field=candidates_field)
        )

    write_jsonl(output_records, output_path)


def _find_candidates_field(record: dict[str, Any]) -> str:
    for field in ("model_output", "model_trans", "mutate_output", "predictions", "outputs"):
        if field in record:
            return field
    raise KeyError("No candidate formula field found")


def _load_ground_truth(
    path: str | None,
    id_field: str,
    formula_field: str,
) -> dict[Any, str]:
    if path is None:
        return {}

    records = read_records(path)
    if isinstance(records, list):
        return {
            record.get(id_field, index): record[formula_field]
            for index, record in enumerate(records)
            if formula_field in record
        }
    return {}
