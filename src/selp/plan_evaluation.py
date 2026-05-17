"""Plan validation against LTL automata."""

from __future__ import annotations

from collections.abc import Iterable

from .ltl import (
    build_full_traj_states,
    get_init_traj_state,
    ltl_to_digraph,
    validate_next_action,
)


def has_in_common(left: set, right: set) -> bool:
    return bool(left & right)


def check_if_plan_can_be_accepted_by_ltl(
    plan: str,
    predicate_set: Iterable[str],
    dfa,
    accepting_states: list[int],
    initial_state: int,
    is_dba: bool,
    check_invalid_predicate: bool = True,
) -> tuple[bool, str]:
    """Check whether a newline-separated plan reaches an accepting automaton state."""

    predicates = set(predicate_set)
    init_traj_state = get_init_traj_state(predicates)
    curr_dfa_state_set = {initial_state}
    accepting_states_set = set(accepting_states)
    plan_steps = [line.strip() for line in plan.splitlines() if line.strip() and line.strip() != "DONE"]

    for index, plan_step in enumerate(plan_steps):
        if check_invalid_predicate and plan_step not in predicates:
            return False, f"invalid param (out of predicate set): {plan_step}"

        if f"!{plan_step}" not in init_traj_state:
            return False, f"invalid plan step: {plan_step}"

        traj_state = init_traj_state.replace(f"!{plan_step}", plan_step)
        new_curr_dfa_state_set: set[int] = set()
        find_valid_flag = False
        for curr_dfa_state in curr_dfa_state_set:
            valid_result, next_state_set = validate_next_action(
                dfa, curr_dfa_state, traj_state, accepting_states, is_dba
            )
            if valid_result:
                find_valid_flag = True
                new_curr_dfa_state_set.update(next_state_set)

        if not find_valid_flag:
            return False, f"invalid step leads to invalid automaton state: {plan_step}"

        curr_dfa_state_set = new_curr_dfa_state_set
        if has_in_common(curr_dfa_state_set, accepting_states_set) and index == len(plan_steps) - 1:
            return True, ""

    return False, "plan does not lead to accepting states"


def evaluate_plan_against_formula(
    plan: str,
    formula: str,
    predicate_set: Iterable[str],
) -> tuple[bool, str]:
    predicates = set(predicate_set)
    dfa, accepting_states, initial_state, is_dba = ltl_to_digraph(
        formula,
        build_full_traj_states(predicates),
    )
    if dfa is None or accepting_states is None or initial_state is None or is_dba is None:
        return False, "formula automaton is empty"
    return check_if_plan_can_be_accepted_by_ltl(
        plan, predicates, dfa, accepting_states, initial_state, is_dba
    )
