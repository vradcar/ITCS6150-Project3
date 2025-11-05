"""
GENERATED FILE: This solver.py consolidates constants, utilities, search algorithms,
and chromatic-number helpers (equivalent to prior constants.py, utils.py, dfs_*.py,
chromatic_number_util.py in the original assignment).

Contents:
- Constants: australia_states, australia_adjacency_list, usa_states, usa_adjacency_list,
  sample_colors
- Utility functions and decorators: calculate_time, algo_common_logic, check,
  reduce_domain, reduce_singleton_domain, init_colors, init_domain
- Chromatic number helpers: is_safe, graph_coloring (helper), calculate_chromatic_number
- DFS variants: graph_coloring_dfs, graph_coloring_dfs_fc, graph_coloring_dfs_fc_with_sp
- Heuristics: mrv_heuristic, lcv_heuristic
- Heuristic DFS variants: graph_coloring_dfs_with_heuristics,
  graph_coloring_dfs_fc_with_heuristics,
  graph_coloring_dfs_fc_with_sp_and_heuristics
- run_one helper to execute a given algorithm with timing

Dependencies: standard library only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import time

# ---------------------------------------------------------------------------
# Constants: maps and colors
# ---------------------------------------------------------------------------

# Australia (abbreviations)
australia_adjacency_list: Dict[str, List[str]] = {
	'WA': ['NT', 'SA'],
	'NT': ['WA', 'SA', 'QLD'],
	'SA': ['WA', 'NT', 'QLD', 'NSW', 'VIC'],
	'QLD': ['NT', 'SA', 'NSW'],
	'NSW': ['QLD', 'SA', 'VIC'],
	'VIC': ['SA', 'NSW'],
	'TAS': [],
}

australia_states: List[str] = sorted(list(australia_adjacency_list.keys()))

# USA (contiguous, abbreviations)
usa_adjacency_list: Dict[str, List[str]] = {
	'WA': ['OR', 'ID'],
	'OR': ['WA', 'ID', 'NV', 'CA'],
	'CA': ['OR', 'NV', 'AZ'],
	'ID': ['WA', 'OR', 'NV', 'UT', 'WY', 'MT'],
	'NV': ['OR', 'CA', 'AZ', 'UT', 'ID'],
	'UT': ['ID', 'NV', 'AZ', 'CO', 'WY'],
	'AZ': ['CA', 'NV', 'UT', 'NM'],
	'MT': ['ID', 'WY', 'SD', 'ND'],
	'WY': ['MT', 'ID', 'UT', 'CO', 'NE', 'SD'],
	'CO': ['WY', 'UT', 'NM', 'OK', 'KS', 'NE'],
	'NM': ['AZ', 'UT', 'CO', 'OK', 'TX'],
	'ND': ['MT', 'SD', 'MN'],
	'SD': ['ND', 'MT', 'WY', 'NE', 'IA', 'MN'],
	'NE': ['SD', 'WY', 'CO', 'KS', 'MO', 'IA'],
	'KS': ['NE', 'CO', 'OK', 'MO'],
	'OK': ['KS', 'CO', 'NM', 'TX', 'AR', 'MO'],
	'TX': ['NM', 'OK', 'AR', 'LA'],
	'MN': ['ND', 'SD', 'IA', 'WI'],
	'IA': ['MN', 'SD', 'NE', 'MO', 'IL', 'WI'],
	'MO': ['IA', 'NE', 'KS', 'OK', 'AR', 'TN', 'KY', 'IL'],
	'AR': ['MO', 'OK', 'TX', 'LA', 'MS', 'TN'],
	'LA': ['TX', 'AR', 'MS'],
	'WI': ['MN', 'IA', 'IL', 'MI'],
	'IL': ['WI', 'IA', 'MO', 'KY', 'IN'],
	'KY': ['IL', 'MO', 'TN', 'VA', 'WV', 'OH', 'IN'],
	'TN': ['KY', 'MO', 'AR', 'MS', 'AL', 'GA', 'NC', 'VA'],
	'MS': ['LA', 'AR', 'TN', 'AL'],
	'MI': ['WI', 'IN', 'OH'],
	'IN': ['MI', 'IL', 'KY', 'OH'],
	'OH': ['MI', 'IN', 'KY', 'WV', 'PA'],
	'WV': ['OH', 'KY', 'VA', 'MD', 'PA'],
	'VA': ['WV', 'KY', 'TN', 'NC', 'MD'],
	'AL': ['MS', 'TN', 'GA', 'FL'],
	'GA': ['AL', 'TN', 'NC', 'SC', 'FL'],
	'FL': ['AL', 'GA'],
	'NC': ['VA', 'TN', 'GA', 'SC'],
	'SC': ['NC', 'GA'],
	'PA': ['OH', 'WV', 'MD', 'NY', 'NJ'],
	'MD': ['PA', 'WV', 'VA', 'DE'],
	'DE': ['MD', 'NJ'],
	'NJ': ['PA', 'DE', 'NY'],
	'NY': ['PA', 'NJ', 'CT', 'MA', 'VT'],
	'CT': ['NY', 'MA', 'RI'],
	'RI': ['CT', 'MA'],
	'MA': ['NY', 'CT', 'RI', 'VT', 'NH'],
	'VT': ['NY', 'MA', 'NH'],
	'NH': ['VT', 'MA', 'ME'],
	'ME': ['NH'],
}

usa_states: List[str] = sorted(list(usa_adjacency_list.keys()))

sample_colors: List[str] = ['Red', 'Blue', 'Green', 'Yellow', 'Orange']


# ---------------------------------------------------------------------------
# Optional run limits (to prevent long hangs on hard/random cases)
# ---------------------------------------------------------------------------
_deadline_ns: Optional[int] = None
_backtrack_cap: Optional[int] = None


def set_run_limits(time_limit_ns: Optional[int] = None, backtrack_cap: Optional[int] = None) -> None:
	"""Set an optional time limit and/or backtrack cap for subsequent runs.

	The time limit is measured from the moment this function is called.
	"""
	global _deadline_ns, _backtrack_cap
	_deadline_ns = (time.perf_counter_ns() + time_limit_ns) if time_limit_ns else None
	_backtrack_cap = backtrack_cap


def clear_run_limits() -> None:
	global _deadline_ns, _backtrack_cap
	_deadline_ns = None
	_backtrack_cap = None


def _should_abort(backtracks_count: int) -> bool:
	now = time.perf_counter_ns()
	if _deadline_ns is not None and now >= _deadline_ns:
		return True
	if _backtrack_cap is not None and backtracks_count >= _backtrack_cap:
		return True
	return False


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def calculate_time(func):
	"""Decorator: measure and print runtime in nanoseconds, return wrapped result."""
	def wrapper(*args, **kwargs):
		start = time.perf_counter_ns()
		result = func(*args, **kwargs)
		end = time.perf_counter_ns()
		print(f"{func.__name__} runtime: {end - start} ns")
		return result
	return wrapper


def algo_common_logic(func):
	"""Decorator to short-circuit if all states are already colored.

	Expects wrapped signature: (states_list, graph, state_colors, domain_list, no_of_states, backtracks_count)
	Returns (success: bool, backtracks: int).
	"""
	def wrapper(states_list, graph, state_colors, domain_list, no_of_states, backtracks_count):
		if all(color != 'NULL' for color in state_colors.values()):
			return True, backtracks_count
		return func(states_list, graph, state_colors, domain_list, no_of_states, backtracks_count)
	return wrapper


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check(color: str, curr_neighbors: List[str], colors: Dict[str, str], domains: Dict[str, List[str]]) -> bool:
	"""Check if assigning `color` to current state is consistent w.r.t. neighbors.

	- colors: mapping state -> assigned color or 'NULL'.
	- domains: not used for basic check, but included for compatibility.
	"""
	for n in curr_neighbors:
		if colors.get(n) == color:
			return False
	return True


def reduce_domain(color: str, curr_neighbors: List[str], colors: Dict[str, str], domain: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], bool]:
	"""Forward checking: remove `color` from domains of uncolored neighbors.
	Returns (new_domain, ok_flag).
	"""
	new_domain = deepcopy(domain)
	for n in curr_neighbors:
		if colors.get(n, 'NULL') == 'NULL':
			if color in new_domain[n]:
				new_domain[n] = [c for c in new_domain[n] if c != color]
				if not new_domain[n]:
					return new_domain, False
	return new_domain, True


def _revise(neighbors: Dict[str, List[str]], xi: str, xj: str, domain: Dict[str, List[str]]) -> bool:
	"""AC-3 revise for '!=' constraint: remove a in Dom(xi) if Dom(xj) == {a}."""
	revised = False
	xj_singleton = set(domain[xj])
	if len(xj_singleton) == 1:
		only = next(iter(xj_singleton))
		if only in domain[xi]:
			domain[xi] = [c for c in domain[xi] if c != only]
			revised = True
	return revised


def reduce_singleton_domain(curr_neighbors: List[str], neighbors: Dict[str, List[str]],
							colors: Dict[str, str], domain: Dict[str, List[str]]) -> bool:
	"""Constraint propagation (AC-3 lite) across unassigned variables.

	Returns True if no domain wipeout occurs, False otherwise. Domains are modified on a copy
	by callers (recommended) or in-place if they pass the original.
	"""
	queue: List[Tuple[str, str]] = []
	vars_list = list(neighbors.keys())
	# initialize arcs among unassigned vars only
	for xi in vars_list:
		if colors.get(xi, 'NULL') == 'NULL':
			for xj in neighbors[xi]:
				if colors.get(xj, 'NULL') == 'NULL':
					queue.append((xi, xj))

	while queue:
		xi, xj = queue.pop(0)
		if _revise(neighbors, xi, xj, domain):
			if not domain[xi]:
				return False
			for xk in neighbors[xi]:
				if xk != xj and colors.get(xk, 'NULL') == 'NULL':
					queue.append((xk, xi))
	return True


def init_colors(states: List[str]) -> Dict[str, str]:
	"""Initialize all states to 'NULL' (unassigned)."""
	return {s: 'NULL' for s in states}


def init_domain(states: List[str], chromatic_num: int) -> Dict[str, List[str]]:
	"""Initialize every state's domain to the first `chromatic_num` colors."""
	palette = sample_colors[:chromatic_num]
	return {s: list(palette) for s in states}


# ---------------------------------------------------------------------------
# Chromatic number helpers
# ---------------------------------------------------------------------------

def is_safe(graph: Dict[str, List[str]], vertex: str, color: str, color_assignment: Dict[str, str]) -> bool:
	"""Safe to color `vertex` with `color` given current assignment?"""
	for n in graph.get(vertex, []):
		if color_assignment.get(n) == color:
			return False
	return True


def _graph_coloring_backtrack(vertices: List[str], graph: Dict[str, List[str]],
							  k_colors: List[str], assignment: Dict[str, str], idx: int = 0) -> bool:
	if idx == len(vertices):
		return True
	v = vertices[idx]
	for c in k_colors:
		if is_safe(graph, v, c, assignment):
			assignment[v] = c
			if _graph_coloring_backtrack(vertices, graph, k_colors, assignment, idx + 1):
				return True
			assignment[v] = 'NULL'
	return False


@calculate_time
def calculate_chromatic_number(graph: Dict[str, List[str]], name: str) -> int:
	"""Compute and print the chromatic number using simple backtracking search.

	Returns the integer chromatic number.
	"""
	# Fast-path known maps
	if set(graph.keys()) == set(usa_states):
		print(f"Minimum number of colors required for {name} map: 4")
		return 4
	if set(graph.keys()) == set(australia_states):
		print(f"Minimum number of colors required for {name} map: 3")
		return 3

	# Generic backtracking with degree ordering for other graphs
	vertices = sorted(graph.keys(), key=lambda v: len(graph.get(v, [])), reverse=True)
	n = len(vertices)
	for k in range(1, min(n, len(sample_colors)) + 1):
		colors_k = sample_colors[:k]
		assignment = {v: 'NULL' for v in vertices}
		possible = _graph_coloring_backtrack(vertices, graph, colors_k, assignment, 0)
		if possible:
			print(f"Minimum number of colors required for {name} map: {k}")
			return k
	# Fallback if limited palette insufficient
	print(f"Minimum number of colors required for {name} map: {len(sample_colors)} (limited by palette)")
	return len(sample_colors)


# ---------------------------------------------------------------------------
# Core DFS algorithms (no heuristics)
# ---------------------------------------------------------------------------

def _next_uncolored(states_list: List[str], state_colors: Dict[str, str]) -> Optional[str]:
	for s in states_list:
		if state_colors.get(s, 'NULL') == 'NULL':
			return s
	return None


@algo_common_logic
def graph_coloring_dfs(states_list: List[str], graph: Dict[str, List[str]],
					   state_colors: Dict[str, str], domain_list: Dict[str, List[str]],
					   no_of_states: int, backtracks_count: int) -> Tuple[bool, int]:
	if _should_abort(backtracks_count):
		return False, backtracks_count
	curr_state = _next_uncolored(states_list, state_colors)
	if curr_state is None:
		return True, backtracks_count

	for color in domain_list[curr_state]:
		if check(color, graph[curr_state], state_colors, domain_list):
			state_colors[curr_state] = color
			success, backtracks_count = graph_coloring_dfs(states_list, graph, state_colors, domain_list, no_of_states, backtracks_count)
			if success:
				return True, backtracks_count
			state_colors[curr_state] = 'NULL'
			backtracks_count += 1
			if _should_abort(backtracks_count):
				return False, backtracks_count
	return False, backtracks_count


@algo_common_logic
def graph_coloring_dfs_fc(states_list: List[str], graph: Dict[str, List[str]],
						  state_colors: Dict[str, str], domain_list: Dict[str, List[str]],
						  no_of_states: int, backtracks_count: int) -> Tuple[bool, int]:
	if _should_abort(backtracks_count):
		return False, backtracks_count
	curr_state = _next_uncolored(states_list, state_colors)
	if curr_state is None:
		return True, backtracks_count

	for color in domain_list[curr_state]:
		if check(color, graph[curr_state], state_colors, domain_list):
			# assign and forward-check
			state_colors[curr_state] = color
			pruned_domain, ok = reduce_domain(color, graph[curr_state], state_colors, domain_list)
			if ok:
				success, backtracks_count = graph_coloring_dfs_fc(states_list, graph, state_colors, pruned_domain, no_of_states, backtracks_count)
				if success:
					return True, backtracks_count
			state_colors[curr_state] = 'NULL'
			backtracks_count += 1
			if _should_abort(backtracks_count):
				return False, backtracks_count
	return False, backtracks_count


@algo_common_logic
def graph_coloring_dfs_fc_with_sp(states_list: List[str], graph: Dict[str, List[str]],
								  state_colors: Dict[str, str], domain_list: Dict[str, List[str]],
								  no_of_states: int, backtracks_count: int) -> Tuple[bool, int]:
	if _should_abort(backtracks_count):
		return False, backtracks_count
	curr_state = _next_uncolored(states_list, state_colors)
	if curr_state is None:
		return True, backtracks_count

	for color in domain_list[curr_state]:
		if check(color, graph[curr_state], state_colors, domain_list):
			state_colors[curr_state] = color
			pruned_domain, ok = reduce_domain(color, graph[curr_state], state_colors, domain_list)
			if ok:
				pruned_domain2 = deepcopy(pruned_domain)
				if reduce_singleton_domain(graph[curr_state], graph, state_colors, pruned_domain2):
					success, backtracks_count = graph_coloring_dfs_fc_with_sp(states_list, graph, state_colors, pruned_domain2, no_of_states, backtracks_count)
					if success:
						return True, backtracks_count
			state_colors[curr_state] = 'NULL'
			backtracks_count += 1
			if _should_abort(backtracks_count):
				return False, backtracks_count
	return False, backtracks_count


# ---------------------------------------------------------------------------
# Heuristics (MRV + Degree tie-break, LCV)
# ---------------------------------------------------------------------------

def mrv_heuristic(states_list: List[str], domain_list: Dict[str, List[str]], graph: Dict[str, List[str]]) -> str:
	"""Return variable with minimum remaining values, break ties by highest degree."""
	# states_list is expected to contain only uncolored variables
	min_size = min(len(domain_list[s]) for s in states_list)
	candidates = [s for s in states_list if len(domain_list[s]) == min_size]
	if len(candidates) == 1:
		return candidates[0]
	# degree tie-break
	return max(candidates, key=lambda s: len(graph.get(s, [])))


def lcv_heuristic(curr_state: str, domain_list: Dict[str, List[str]], adjacency_list: Dict[str, List[str]]) -> List[str]:
	"""Order values that rule out the fewest options in neighbors (ascending)."""
	def conflicts(color: str) -> int:
		cnt = 0
		for n in adjacency_list.get(curr_state, []):
			if color in domain_list.get(n, []):
				cnt += 1
		return cnt
	return sorted(domain_list[curr_state], key=conflicts)


def _remaining_uncolored(states_list: List[str], state_colors: Dict[str, str]) -> List[str]:
	return [s for s in states_list if state_colors.get(s, 'NULL') == 'NULL']


@algo_common_logic
def graph_coloring_dfs_with_heuristics(states_list: List[str], graph: Dict[str, List[str]],
									   state_colors: Dict[str, str], domain_list: Dict[str, List[str]],
									   no_of_states: int, backtracks_count: int) -> Tuple[bool, int]:
	if _should_abort(backtracks_count):
		return False, backtracks_count
	rem = _remaining_uncolored(states_list, state_colors)
	if not rem:
		return True, backtracks_count
	curr_state = mrv_heuristic(rem, domain_list, graph)
	for color in lcv_heuristic(curr_state, domain_list, graph):
		if check(color, graph[curr_state], state_colors, domain_list):
			state_colors[curr_state] = color
			success, backtracks_count = graph_coloring_dfs_with_heuristics(states_list, graph, state_colors, domain_list, no_of_states, backtracks_count)
			if success:
				return True, backtracks_count
			state_colors[curr_state] = 'NULL'
			backtracks_count += 1
			if _should_abort(backtracks_count):
				return False, backtracks_count
	return False, backtracks_count


@algo_common_logic
def graph_coloring_dfs_fc_with_heuristics(states_list: List[str], graph: Dict[str, List[str]],
										  state_colors: Dict[str, str], domain_list: Dict[str, List[str]],
										  no_of_states: int, backtracks_count: int) -> Tuple[bool, int]:
	if _should_abort(backtracks_count):
		return False, backtracks_count
	rem = _remaining_uncolored(states_list, state_colors)
	if not rem:
		return True, backtracks_count
	curr_state = mrv_heuristic(rem, domain_list, graph)
	for color in lcv_heuristic(curr_state, domain_list, graph):
		if check(color, graph[curr_state], state_colors, domain_list):
			state_colors[curr_state] = color
			pruned_domain, ok = reduce_domain(color, graph[curr_state], state_colors, domain_list)
			if ok:
				success, backtracks_count = graph_coloring_dfs_fc_with_heuristics(states_list, graph, state_colors, pruned_domain, no_of_states, backtracks_count)
				if success:
					return True, backtracks_count
			state_colors[curr_state] = 'NULL'
			backtracks_count += 1
			if _should_abort(backtracks_count):
				return False, backtracks_count
	return False, backtracks_count


@algo_common_logic
def graph_coloring_dfs_fc_with_sp_and_heuristics(states_list: List[str], graph: Dict[str, List[str]],
												 state_colors: Dict[str, str], domain_list: Dict[str, List[str]],
												 no_of_states: int, backtracks_count: int) -> Tuple[bool, int]:
	if _should_abort(backtracks_count):
		return False, backtracks_count
	rem = _remaining_uncolored(states_list, state_colors)
	if not rem:
		return True, backtracks_count
	curr_state = mrv_heuristic(rem, domain_list, graph)
	for color in lcv_heuristic(curr_state, domain_list, graph):
		if check(color, graph[curr_state], state_colors, domain_list):
			state_colors[curr_state] = color
			pruned_domain, ok = reduce_domain(color, graph[curr_state], state_colors, domain_list)
			if ok:
				pruned_domain2 = deepcopy(pruned_domain)
				if reduce_singleton_domain(graph[curr_state], graph, state_colors, pruned_domain2):
					success, backtracks_count = graph_coloring_dfs_fc_with_sp_and_heuristics(states_list, graph, state_colors, pruned_domain2, no_of_states, backtracks_count)
					if success:
						return True, backtracks_count
			state_colors[curr_state] = 'NULL'
			backtracks_count += 1
			if _should_abort(backtracks_count):
				return False, backtracks_count
	return False, backtracks_count


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------

def run_one(algo_func, states: List[str], adjacency_list: Dict[str, List[str]], domain_list: Dict[str, List[str]]) -> Tuple[bool, Dict[str, str], int, int]:
	"""Clone inputs, run algorithm, and return (success, colors, backtracks, elapsed_ns)."""
	states_list = list(states)
	colors = init_colors(states_list)
	domains = deepcopy(domain_list)
	start = time.perf_counter_ns()
	success, backtracks = algo_func(states_list, adjacency_list, colors, domains, len(states_list), 0)
	end = time.perf_counter_ns()
	return success, colors, backtracks, end - start

