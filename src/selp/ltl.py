"""LTL and automaton helpers.

The semantic operations use Spot when available. Importing this module does not
require Spot; functions that need it raise a clear error if the package is
missing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _spot():
    try:
        import spot  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Spot is required for semantic LTL operations. Install it with "
            "`conda install -c conda-forge spot`."
        ) from exc
    return spot


def parse_formula(formula: str):
    return _spot().formula(formula)


def is_valid_formula(formula: str) -> bool:
    try:
        parse_formula(formula)
    except Exception:
        return False
    return True


def are_equivalent(left: str, right: str) -> bool:
    spot = _spot()
    return bool(spot.are_equivalent(spot.formula(left), spot.formula(right)))


def atomic_props(formula: str) -> set[str]:
    spot = _spot()
    return {str(prop) for prop in spot.atomic_prop_collect(formula)}


def get_init_traj_state(predicate_set: Iterable[str], init_predicate: str | None = None) -> str:
    predicates = sorted(set(predicate_set))
    if not predicates:
        raise ValueError("predicate_set must not be empty")
    traj_state = "&".join(f"!{predicate}" for predicate in predicates)
    if init_predicate:
        traj_state = traj_state.replace(f"!{init_predicate}", init_predicate)
    return traj_state


def build_full_traj_states(predicate_set: Iterable[str]) -> list[str]:
    predicates = sorted(set(predicate_set))
    init_traj_state = get_init_traj_state(predicates)
    return [init_traj_state.replace(f"!{predicate}", predicate) for predicate in predicates]


def build_hard_constraint(predicate_set: Iterable[str]) -> str:
    predicates = sorted(set(predicate_set))
    init_traj_state = get_init_traj_state(predicates)
    constraints = []
    for predicate in predicates:
        traj_state = init_traj_state.replace(f"!{predicate}", predicate)
        constraints.append(f"(G({predicate}->({traj_state})))")
    return "&".join(constraints)


def _edge_is_feasible(edge_formula: str, possible_traj_list: Iterable[str]) -> bool:
    spot = _spot()
    edge = spot.formula(edge_formula)
    for traj_state in possible_traj_list:
        traj = spot.formula(traj_state)
        if spot.contains(edge, traj) or spot.are_equivalent(edge, traj):
            return True
    return False


def ltl_to_digraph(
    formula: str,
    possible_traj_list: Iterable[str] | None = None,
    hard_constraint: str | None = None,
    automaton=None,
    force_check_feasible_edge: bool = False,
) -> tuple[object | None, list[int] | None, int | None, bool | None]:
    """Convert an LTL formula to a NetworkX graph.

    The source implementation accepted `hard_constraint` but did not conjoin it
    inside this function. This release keeps that behavior for compatibility.
    """

    del hard_constraint
    spot = _spot()
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required for LTL automaton graph conversion") from exc

    aut = automaton if automaton is not None else spot.translate(
        formula, "complete", "state-based", "deterministic"
    )
    if aut.is_empty():
        return None, None, None, None

    bdd = aut.get_dict()
    initial_state = aut.get_init_state_number()
    accepting_states = [
        state for state in range(aut.num_states()) if aut.state_is_accepting(state)
    ]
    is_dba = aut.is_deterministic()

    node_list: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
    trajs = list(possible_traj_list or [])
    for state in range(aut.num_states()):
        for edge in aut.out(state):
            edge_formula = spot.bdd_format_formula(bdd, edge.cond)
            if force_check_feasible_edge and not _edge_is_feasible(edge_formula, trajs):
                continue
            node_list[state][edge.dst] = {"edge_label": edge_formula}

    return nx.DiGraph(node_list), accepting_states, initial_state, is_dba


def progress_ltl(
    dfa,
    curr_dfa_state: int,
    action: str,
    is_dba: bool,
    visited_set: set[int] | None = None,
) -> set[int]:
    spot = _spot()
    action_formula = spot.formula(action)

    if is_dba:
        for next_state in dfa.adj[curr_dfa_state]:
            edge_formula = spot.formula(
                dfa.get_edge_data(curr_dfa_state, next_state, default=0)["edge_label"]
            )
            if spot.contains(edge_formula, action_formula) or spot.are_equivalent(
                edge_formula, action_formula
            ):
                if visited_set is None or next_state not in visited_set:
                    return {next_state}
        return {curr_dfa_state}

    next_state_set: set[int] = set()
    for next_state in dfa.adj[curr_dfa_state]:
        edge_formula = spot.formula(
            dfa.get_edge_data(curr_dfa_state, next_state, default=0)["edge_label"]
        )
        if spot.contains(edge_formula, action_formula) or spot.are_equivalent(
            edge_formula, action_formula
        ):
            if visited_set is None or next_state not in visited_set:
                next_state_set.add(next_state)

    return next_state_set if next_state_set else {curr_dfa_state}


def validate_next_action(
    dfa,
    curr_dfa_state: int,
    traj_state: str,
    accepting_states: list[int],
    is_dba: bool,
) -> tuple[bool, set[int]]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required for LTL plan validation") from exc

    if traj_state == "stop":
        return curr_dfa_state in accepting_states, {curr_dfa_state}

    valid_next_states: set[int] = set()
    next_state_set = progress_ltl(dfa, curr_dfa_state, traj_state, is_dba)
    for next_state in next_state_set:
        for accepting_state in accepting_states:
            if nx.has_path(dfa, next_state, accepting_state) or next_state == accepting_state:
                valid_next_states.add(next_state)

    if valid_next_states:
        return True, valid_next_states
    return False, next_state_set
