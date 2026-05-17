"""Trie-based constrained decoding for SELP plan generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .ltl import get_init_traj_state, validate_next_action
from .plan_evaluation import has_in_common


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    is_end_of_word: bool = False


def insert(root: TrieNode, key: Iterable[int]) -> None:
    node = root
    for token in key:
        node = node.children.setdefault(token, TrieNode())
    node.is_end_of_word = True


def search(root: TrieNode, key: Iterable[int]) -> bool:
    node = root
    for token in key:
        if token not in node.children:
            return False
        node = node.children[token]
    return node.is_end_of_word


def remove(root: TrieNode | None, key: list[int], depth: int = 0) -> TrieNode | None:
    if root is None:
        return None

    if depth == len(key):
        root.is_end_of_word = False
        if not root.children:
            return None
        return root

    token = key[depth]
    if token not in root.children:
        return root

    child = remove(root.children[token], key, depth + 1)
    if child is None:
        del root.children[token]
    else:
        root.children[token] = child

    if not root.children and not root.is_end_of_word:
        return None
    return root


def build_trie(token_sequences: Iterable[Iterable[int]]) -> TrieNode:
    root = TrieNode()
    for sequence in token_sequences:
        insert(root, sequence)
    return root


def token_ids_for_predicates(predicate_set: Iterable[str], tokenizer) -> list[list[int]]:
    token_sequences: list[list[int]] = []
    for predicate in sorted(set(predicate_set)):
        token_ids = list(tokenizer.encode(f"{predicate}\n", add_special_tokens=False))
        if token_ids and tokenizer.decode(token_ids[0], skip_special_tokens=True) == "<s>":
            token_ids = token_ids[1:]
        token_sequences.append(token_ids)
    return token_sequences


def ban_word_from_trie(root: TrieNode, tokenizer, word: str) -> TrieNode:
    token_ids = list(tokenizer.encode(word, add_special_tokens=False))
    if token_ids and tokenizer.decode(token_ids[0], skip_special_tokens=True) == "<s>":
        token_ids = token_ids[1:]
    return remove(root, token_ids) or TrieNode()


def constrained_decode_plan_trie(
    tokenizer,
    model,
    predicate_set: Iterable[str],
    dfa,
    accepting_states: list[int],
    initial_state: int,
    is_dba: bool,
    user_prompt: str,
    temperature: float,
    eos_id: int | None = None,
    pad_id: int | None = None,
    device: int | str = 0,
    max_plan_steps: int = 500,
    max_plan_lines: int = 100,
    num_output: int = 1,
    force_visiting_new_state: bool = False,
    return_at_once: bool = False,
    remove_word_from_trie_list: list[str] | None = None,
    next_token_fn: Callable[[str, list[int]], tuple[str, int]] | None = None,
    validate_action_fn: Callable[[Any, int, str, list[int], bool], tuple[bool, set[int]]] | None = None,
) -> tuple[str, str, int, int]:
    """Generate a plan while pruning tokens that cannot form valid predicates.

    `next_token_fn` and `validate_action_fn` are injectable for unit tests. When
    omitted, Hugging Face generation and Spot-backed validation are used.
    """

    predicates = set(predicate_set)
    if not predicates:
        raise ValueError("predicate_set must not be empty")

    validate_action_fn = validate_action_fn or validate_next_action
    curr_dfa_state_set = {initial_state}
    last_visited_state_set = {initial_state}
    accepting_states_set = set(accepting_states)
    init_traj_state = get_init_traj_state(predicates)
    full_plan = user_prompt
    violate_num = 0

    trie_root = build_trie(token_ids_for_predicates(predicates, tokenizer))
    curr_trie_node = trie_root

    for word in remove_word_from_trie_list or []:
        trie_root = ban_word_from_trie(trie_root, tokenizer, f"{word}\n")
        curr_trie_node = trie_root

    for step_index in range(max_plan_steps):
        if len(full_plan.splitlines()) > max_plan_lines:
            return full_plan, f"plan step exceeds max number: {max_plan_steps}", violate_num, step_index

        content = ""
        while not content.endswith("\n"):
            allowed_token_ids = list(curr_trie_node.children.keys())
            if not allowed_token_ids:
                return full_plan, "no valid token in the trie", violate_num, step_index

            if len(allowed_token_ids) == 1:
                token_id = allowed_token_ids[0]
                response_token = tokenizer.decode(token_id)
            else:
                response_token, token_id = _generate_next_token(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=full_plan,
                    allowed_token_ids=allowed_token_ids,
                    temperature=temperature,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    device=device,
                    num_output=num_output,
                    next_token_fn=next_token_fn,
                )

            curr_trie_node = curr_trie_node.children[token_id]
            content += response_token

        param = content.strip()
        if return_at_once:
            return param, "", violate_num, step_index

        if param not in predicates:
            raise ValueError(f"generated parameter is not in predicate set: {param}")
        if f"!{param}" not in init_traj_state:
            raise ValueError(f"invalid plan step generated: {param}")

        traj_state = init_traj_state.replace(f"!{param}", param)
        find_valid_flag = False
        new_curr_dfa_state_set: set[int] = set()

        for curr_dfa_state in curr_dfa_state_set:
            valid_result, next_valid_state_set = validate_action_fn(
                dfa, curr_dfa_state, traj_state, accepting_states, is_dba
            )
            if not valid_result:
                continue
            if force_visiting_new_state:
                new_curr_dfa_state_set.update(
                    state for state in next_valid_state_set if state not in last_visited_state_set
                )
            else:
                new_curr_dfa_state_set.update(next_valid_state_set)
            find_valid_flag = bool(new_curr_dfa_state_set)

        if not find_valid_flag:
            trie_root = ban_word_from_trie(trie_root, tokenizer, f"{param}\n")
            curr_trie_node = trie_root
            violate_num += 1
            continue

        curr_dfa_state_set = set(new_curr_dfa_state_set)
        if force_visiting_new_state:
            last_visited_state_set.update(curr_dfa_state_set)

        if full_plan and not full_plan.endswith("\n"):
            full_plan += "\n"
        full_plan += content
        if not full_plan.endswith("\n"):
            full_plan += "\n"

        trie_root = build_trie(token_ids_for_predicates(predicates, tokenizer))
        trie_root = ban_word_from_trie(trie_root, tokenizer, f"{param}\n")
        curr_trie_node = trie_root

        if has_in_common(curr_dfa_state_set, accepting_states_set):
            return full_plan + "DONE", "", violate_num, step_index + 1

    return full_plan, f"plan step exceeds max number: {max_plan_steps}", violate_num, max_plan_steps


def _generate_next_token(
    tokenizer,
    model,
    prompt: str,
    allowed_token_ids: list[int],
    temperature: float,
    eos_id: int | None,
    pad_id: int | None,
    device: int | str,
    num_output: int,
    next_token_fn: Callable[[str, list[int]], tuple[str, int]] | None,
) -> tuple[str, int]:
    if next_token_fn is not None:
        response_token, token_id = next_token_fn(prompt, allowed_token_ids)
        if token_id not in allowed_token_ids:
            raise ValueError(f"next_token_fn returned disallowed token id {token_id}")
        return response_token, token_id

    try:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required for model decoding") from exc

    class AllowListLogitsProcessor(LogitsProcessor):
        def __init__(self, allow_token_id_list: list[int]):
            self.allow_token_id_list = allow_token_id_list

        def __call__(self, input_ids, scores):
            new_scores = scores.clone().fill_(-float("inf"))
            for token in self.allow_token_id_list:
                new_scores[:, token] = scores[:, token]
            return new_scores

    inputs_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    do_sample = temperature > 0
    generation_kwargs = {
        "do_sample": do_sample,
        "num_return_sequences": num_output,
    }
    if do_sample:
        generation_kwargs.update({"top_k": 0, "top_p": 1.0, "temperature": temperature})
    else:
        generation_kwargs["num_beams"] = max(1, num_output)
    output_ids = model.generate(
        inputs=inputs_ids,
        max_new_tokens=1,
        logits_processor=LogitsProcessorList([AllowListLogitsProcessor(allowed_token_ids)]),
        **generation_kwargs,
        remove_invalid_values=True,
        renormalize_logits=True,
        pad_token_id=pad_id,
        eos_token_id=[eos_id] if eos_id is not None else None,
    )
    torch.cuda.empty_cache()
    generated_token_id = output_ids[0][inputs_ids.size(1):].tolist()[0]
    return tokenizer.decode(generated_token_id), generated_token_id
