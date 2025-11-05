"""
GENERATED FILE: This visualize.py consolidates CLI, visualization, and experiment harness
for the map-coloring CSP assignment (originally spread across multiple files).

Dependencies (install if needed):
	pip install networkx matplotlib prettytable

Minimal usage examples:
	Interactive: python visualize.py
	Batch:       python visualize.py --batch --repeats 5 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as stats
import time
from copy import deepcopy
import json
from urllib.request import urlopen
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from prettytable import PrettyTable

from solver import (
	australia_adjacency_list,
	australia_states,
	usa_adjacency_list,
	usa_states,
	sample_colors,
	calculate_chromatic_number,
	init_domain,
	run_one,
	set_run_limits,
	clear_run_limits,
	graph_coloring_dfs,
	graph_coloring_dfs_fc,
	graph_coloring_dfs_fc_with_sp,
	graph_coloring_dfs_with_heuristics,
	graph_coloring_dfs_fc_with_heuristics,
	graph_coloring_dfs_fc_with_sp_and_heuristics,
)


# Global toggle used by batch harness to randomize state order for non-heuristic runs
_RANDOMIZE_WITHOUT_ORDER = False


# --- Approximate geographic layouts for plotting on "actual" map positions ---
def _australia_positions() -> Dict[str, Tuple[float, float]]:
	# Rough lat/lon-like positions for visualization
	return {
		'WA': (-122, -26),
		'NT': (-133, -19),
		'SA': (-135, -30),
		'QLD': (-146, -21),
		'NSW': (-147, -33),
		'VIC': (-144, -37),
		'TAS': (-147, -42),
	}


def _usa_positions() -> Dict[str, Tuple[float, float]]:
	# Very rough positions approximating contiguous US regions (not exact lon/lat)
	# West Coast
	pos = {
		'WA': (-123, 47), 'OR': (-120, 44), 'CA': (-119, 36),
		# Mountain
		'ID': (-114, 45), 'MT': (-111, 47), 'NV': (-117, 39), 'UT': (-111.5, 40), 'AZ': (-111, 34),
		'WY': (-107.5, 43), 'CO': (-105.5, 39), 'NM': (-106, 34),
		# Plains
		'ND': (-100, 47), 'SD': (-99, 44), 'NE': (-98, 41.5), 'KS': (-97, 38.5), 'OK': (-97.5, 35.5), 'TX': (-99, 31),
		# Midwest / Great Lakes
		'MN': (-94, 46), 'IA': (-93, 42), 'MO': (-92, 38.5), 'WI': (-90.5, 44.5), 'IL': (-89, 40), 'IN': (-86.5, 40),
		'MI': (-84.5, 44.5), 'OH': (-82.5, 40.5),
		# South / Southeast
		'AR': (-92.5, 35), 'LA': (-91.5, 31), 'MS': (-89.5, 32.5), 'AL': (-86.5, 32), 'TN': (-86.5, 36), 'KY': (-85, 38.5),
		'GA': (-83.5, 33), 'FL': (-82, 28), 'SC': (-81, 33.5), 'NC': (-79.5, 35.5), 'VA': (-78.5, 37.5), 'WV': (-80.5, 38.5),
		# Northeast / Mid-Atlantic / New England
		'PA': (-77.5, 41), 'NY': (-75, 43), 'NJ': (-74.8, 40), 'DE': (-75.5, 39), 'MD': (-76.5, 39),
		'CT': (-72.7, 41.6), 'RI': (-71.6, 41.6), 'MA': (-71.8, 42.2), 'VT': (-72.8, 44.2), 'NH': (-71.5, 43.8), 'ME': (-69.2, 45.2),
	}
	return pos


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
	# Monotone chain convex hull
	pts = sorted(set(points))
	if len(pts) <= 1:
		return pts

	def cross(o, a, b):
		return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

	lower = []
	for p in pts:
		while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
			lower.pop()
		lower.append(p)

	upper = []
	for p in reversed(pts):
		while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
			upper.pop()
		upper.append(p)

	return lower[:-1] + upper[:-1]


def _get_map_name_and_positions(graph: Dict[str, List[str]]):
	keys = set(graph.keys())
	if keys == set(usa_adjacency_list.keys()):
		return 'USA', _usa_positions()
	if keys == set(australia_adjacency_list.keys()):
		return 'Australia', _australia_positions()
	return 'Graph', None


# --- GeoJSON drawing for actual state borders (USA/Australia) ---
_US_GEOJSON_URL = (
	"https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
_AUS_GEOJSON_URL = (
	"https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson"
)

_US_ABBR_TO_NAME = {
	'AL': 'Alabama','AZ': 'Arizona','AR': 'Arkansas','CA': 'California','CO': 'Colorado','CT': 'Connecticut',
	'DE': 'Delaware','FL': 'Florida','GA': 'Georgia','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','IA': 'Iowa',
	'KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana','ME': 'Maine','MD': 'Maryland','MA': 'Massachusetts',
	'MI': 'Michigan','MN': 'Minnesota','MS': 'Mississippi','MO': 'Missouri','MT': 'Montana','NE': 'Nebraska',
	'NV': 'Nevada','NH': 'New Hampshire','NJ': 'New Jersey','NM': 'New Mexico','NY': 'New York',
	'NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma','OR': 'Oregon','PA': 'Pennsylvania',
	'RI': 'Rhode Island','SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah',
	'VT': 'Vermont','VA': 'Virginia','WA': 'Washington','WV': 'West Virginia','WI': 'Wisconsin','WY': 'Wyoming'
}

_AUS_ABBR_TO_NAME = {
	'WA': 'Western Australia','NT': 'Northern Territory','SA': 'South Australia','QLD': 'Queensland',
	'NSW': 'New South Wales','VIC': 'Victoria','TAS': 'Tasmania'
}


def _fetch_geojson(url: str):
	with urlopen(url) as resp:
		return json.loads(resp.read().decode('utf-8'))


def _iter_rings(geom):
	gtype = geom['type']
	if gtype == 'Polygon':
		for ring in geom['coordinates']:
			yield ring
	elif gtype == 'MultiPolygon':
		for poly in geom['coordinates']:
			for ring in poly:
				yield ring


def _centroid_of_ring(ring):
	xs = [p[0] for p in ring]
	ys = [p[1] for p in ring]
	return (sum(xs)/len(xs), sum(ys)/len(ys))


def _draw_geojson_map(ax, geojson_data, name_key: str, wanted_name_to_color: Dict[str, str]):
	drawn = False
	for feat in geojson_data.get('features', []):
		props = feat.get('properties', {})
		geom = feat.get('geometry')
		if not geom:
			continue
		name = props.get(name_key)
		if name not in wanted_name_to_color:
			continue
		face = wanted_name_to_color[name]
		# Draw outer rings
		for ring in _iter_rings(geom):
			# ring: [ [lon,lat], ... ]
			xy = [(x, y) for x, y in ring]
			patch = mpatches.Polygon(xy, closed=True, facecolor=face, edgecolor='black', linewidth=0.6, alpha=0.9)
			ax.add_patch(patch)
			drawn = True
	if drawn:
		ax.autoscale()
		ax.set_aspect('equal', adjustable='datalim')
		ax.axis('off')
	return drawn


def visualize_map(graph: Dict[str, List[str]], colored_states: Dict[str, str]) -> None:
	"""Draw the graph with node colors on an approximate geographic layout.

	If known map (USA/Australia), uses pre-defined positions and draws a light
	convex hull as background. Otherwise, falls back to spring_layout.
	"""
	# Try GeoJSON “true borders” rendering for USA/Australia
	map_name, _ = _get_map_name_and_positions(graph)
	if map_name in ('USA', 'Australia'):
		fig, ax = plt.subplots(figsize=(10, 8))
		try:
			if map_name == 'USA':
				gj = _fetch_geojson(_US_GEOJSON_URL)
				name_to_color = {}
				for abbr, color in colored_states.items():
					if abbr in _US_ABBR_TO_NAME:
						name_to_color[_US_ABBR_TO_NAME[abbr]] = color
				drawn = _draw_geojson_map(ax, gj, name_key='name', wanted_name_to_color=name_to_color)
			else:  # Australia
				gj = _fetch_geojson(_AUS_GEOJSON_URL)
				name_to_color = {}
				for abbr, color in colored_states.items():
					if abbr in _AUS_ABBR_TO_NAME:
						name_to_color[_AUS_ABBR_TO_NAME[abbr]] = color
				# The property for state name in this dataset is often 'STATE_NAME'
				drawn = _draw_geojson_map(ax, gj, name_key='STATE_NAME', wanted_name_to_color=name_to_color)
			if drawn:
				ax.set_title(f"{map_name} Map Coloring (state borders)")
				plt.tight_layout()
				plt.show()
				return
		except Exception as e:
			# Fallback to graph plot if fetch/parse fails
			pass

	# Fallback: graph drawing (spring layout or approximate positions)
	G = nx.Graph()
	for u, nbrs in graph.items():
		for v in nbrs:
			G.add_edge(u, v)

	map_name, positions = _get_map_name_and_positions(graph)
	if positions is None:
		pos = nx.spring_layout(G, seed=42)
	else:
		xs = [positions[n][0] for n in G.nodes() if n in positions]
		ys = [positions[n][1] for n in G.nodes() if n in positions]
		minx, maxx = min(xs), max(xs)
		miny, maxy = min(ys), max(ys)
		def norm(p):
			x = (p[0]-minx)/(maxx-minx+1e-9)
			y = (p[1]-miny)/(maxy-miny+1e-9)
			return (x, 1.0 - y)
		pos = {n: norm(positions[n]) for n in G.nodes() if n in positions}

	node_colors = [colored_states.get(n, 'white') for n in G.nodes()]
	plt.figure(figsize=(10, 8))
	nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.0)
	nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, edgecolors='black')
	nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
	plt.axis('off')
	plt.tight_layout()
	plt.show()


def display_table(data: Dict[str, str]) -> None:
	"""Print a state->color mapping using PrettyTable."""
	table = PrettyTable()
	table.field_names = ["State", "Color"]
	for k in sorted(data.keys()):
		table.add_row([k, data[k]])
	print(table)


def get_required_data(choice: int):
	"""Return (states, adjacency_list, color_list, domain_list) for the selected country.

	choice: 1 for USA, 2 for Australia.
	"""
	if choice == 1:
		name = "USA"
		states = deepcopy(usa_states)
		adjacency_list = deepcopy(usa_adjacency_list)
	elif choice == 2:
		name = "Australia"
		states = deepcopy(australia_states)
		adjacency_list = deepcopy(australia_adjacency_list)
	else:
		raise ValueError("Invalid country choice. Use 1 for USA or 2 for Australia.")

	chromatic_number = calculate_chromatic_number(adjacency_list, name)
	domain_list = init_domain(states, chromatic_number)
	return states, adjacency_list, sample_colors[:chromatic_number], domain_list


def _algo_from_choices(heuristic_choice: int, method_choice: int):
	"""Map menu choices to the appropriate solver function."""
	with_heuristics = (heuristic_choice == 2)
	if method_choice == 1:
		return graph_coloring_dfs_with_heuristics if with_heuristics else graph_coloring_dfs
	elif method_choice == 2:
		return graph_coloring_dfs_fc_with_heuristics if with_heuristics else graph_coloring_dfs_fc
	elif method_choice == 3:
		return (
			graph_coloring_dfs_fc_with_sp_and_heuristics if with_heuristics else graph_coloring_dfs_fc_with_sp
		)
	else:
		raise ValueError("Invalid method choice. Use 1, 2, or 3.")


def exec_algo_from_choice(heuristic_choice: int, method_choice: int, country_choice_selected: int, do_visualize: bool = True, quiet: bool = False, time_limit_ns: int | None = None, backtrack_cap: int | None = None):
	"""Run the chosen algorithm and print/visualize results.

	Returns a tuple: (success: bool, colored_states: dict, total_backtracks: int, elapsed_ns: int)
	"""
	states, adjacency_list, color_list, domain_list = get_required_data(country_choice_selected)

	# Randomize order for non-heuristic runs when batch harness toggles the flag
	if _RANDOMIZE_WITHOUT_ORDER and heuristic_choice == 1:
		random.shuffle(states)

	algo_func = _algo_from_choices(heuristic_choice, method_choice)
	if time_limit_ns or backtrack_cap:
		set_run_limits(time_limit_ns=time_limit_ns, backtrack_cap=backtrack_cap)
	try:
		success, colors, backtracks, elapsed_ns = run_one(algo_func, states, adjacency_list, domain_list)
	finally:
		if time_limit_ns or backtrack_cap:
			clear_run_limits()

	if not quiet:
		print("\nResult coloring:")
		display_table(colors)
		print(f"Total backtracks: {backtracks}")
		print(f"Elapsed time: {elapsed_ns} ns")

	if do_visualize:
		visualize_map(adjacency_list, colors)

	return success, colors, backtracks, elapsed_ns


def run_all(method_choice: int = 3, logfile: str | None = None):
	"""Run both USA and Australia, for both heuristic modes, visualize and log results."""
	def log(msg: str):
		print(msg)
		if logfile:
			with open(logfile, 'a', encoding='utf-8') as f:
				f.write(msg + "\n")

	scenarios = [(1, 'USA'), (2, 'Australia')]
	for country_choice, name in scenarios:
		for heuristic_choice in (1, 2):  # without, with
			label = 'with heuristics' if heuristic_choice == 2 else 'without heuristics'
			log(f"\n=== {name} — Method {method_choice} — {label} ===")
			success, colors, backtracks, elapsed_ns = exec_algo_from_choice(
				heuristic_choice, method_choice, country_choice_selected=country_choice, do_visualize=True
			)
			# Persist a compact snapshot too
			log(f"success={success}, backtracks={backtracks}, elapsed_ns={elapsed_ns}")
			for s in sorted(colors):
				log(f"{s}: {colors[s]}")


def _prompt_int(prompt: str, valid: List[int]) -> int:
	while True:
		try:
			val = int(input(prompt).strip())
			if val in valid:
				return val
		except Exception:
			pass
		print(f"Please enter one of: {valid}")


def interactive_cli():
	print("\nMap Coloring using CSP — Interactive Mode\n")
	print("Choose a country:")
	print("  1. USA")
	print("  2. Australia")
	country_choice = _prompt_int("Enter choice (1-2): ", [1, 2])

	print("\nRun with heuristics?")
	print("  1. Without heuristics (fixed/random order)")
	print("  2. With heuristics (MRV + Degree, LCV)")
	heuristic_choice = _prompt_int("Enter choice (1-2): ", [1, 2])

	print("\nChoose a method:")
	print("  1. DFS only")
	print("  2. DFS + Forward Checking")
	print("  3. DFS + Forward Checking + Singleton Propagation")
	method_choice = _prompt_int("Enter choice (1-3): ", [1, 2, 3])

	exec_algo_from_choice(heuristic_choice, method_choice, country_choice_selected=country_choice, do_visualize=True)


def run_batch(repeats: int = 5, randomize_without: bool = False, output_csv: str | None = None, time_limit_ms: int | None = None, backtrack_cap: int | None = None):
	"""Run batch experiments and print summary stats. Optionally save per-trial CSV.

	For each map in {USA, Australia}, each heuristic mode in {without, with}, and each
	method in {1..3}, repeat `repeats` times. When running without heuristics, the
	state order is randomized if `randomize_without` is True.
	"""
	global _RANDOMIZE_WITHOUT_ORDER

	combinations = []
	countries = {1: "USA", 2: "Australia"}
	rows = []  # per-trial

	for country_choice in [1, 2]:
		for heuristic_choice in [1, 2]:  # without, with
			for method_choice in [1, 2, 3]:
				combinations.append((country_choice, heuristic_choice, method_choice))

	for (country_choice, heuristic_choice, method_choice) in combinations:
		label = (
			countries[country_choice],
			"with" if heuristic_choice == 2 else "without",
			method_choice,
		)
		print(f"\n=== Running: map={label[0]}, heuristics={label[1]}, method={label[2]} ===")

		# Toggle randomization for non-heuristic runs
		_RANDOMIZE_WITHOUT_ORDER = (heuristic_choice == 1 and randomize_without)

		for t in range(1, repeats + 1):
			success, _, backtracks, elapsed_ns = exec_algo_from_choice(
				heuristic_choice, method_choice, country_choice_selected=country_choice, do_visualize=False, quiet=True,
				time_limit_ns=(time_limit_ms * 1_000_000) if (time_limit_ms and (heuristic_choice == 1)) else None,
				backtrack_cap=backtrack_cap if (heuristic_choice == 1) else None
			)
			rows.append({
				'map': label[0],
				'heuristic': label[1],
				'method': method_choice,
				'trial': t,
				'backtracks': backtracks,
				'elapsed_ns': elapsed_ns,
				'success': success,
			})

	# Reset global flag
	_RANDOMIZE_WITHOUT_ORDER = False

	# Summarize
	print("\nSummary (average and stddev over successful trials):")
	summary = {}
	for r in rows:
		if not r['success']:
			continue
		key = (r['map'], r['heuristic'], r['method'])
		summary.setdefault(key, {'bt': [], 'ns': []})
		summary[key]['bt'].append(r['backtracks'])
		summary[key]['ns'].append(r['elapsed_ns'])

	table = PrettyTable()
	table.field_names = ["Map", "Heuristic", "Method", "Avg Backtracks", "Std Backtracks", "Avg ns", "Std ns"]
	for key, agg in sorted(summary.items()):
		bt = agg['bt']
		ns = agg['ns']
		avg_bt = stats.mean(bt) if bt else float('nan')
		std_bt = stats.stdev(bt) if len(bt) > 1 else 0.0
		avg_ns = stats.mean(ns) if ns else float('nan')
		std_ns = stats.stdev(ns) if len(ns) > 1 else 0.0
		table.add_row([key[0], key[1], key[2], f"{avg_bt:.2f}", f"{std_bt:.2f}", f"{avg_ns:.0f}", f"{std_ns:.0f}"])
	print(table)

	if output_csv:
		with open(output_csv, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=['map', 'heuristic', 'method', 'trial', 'backtracks', 'elapsed_ns', 'success'])
			writer.writeheader()
			for r in rows:
				writer.writerow(r)
		print(f"Saved per-trial results to {output_csv}")


def main():
	parser = argparse.ArgumentParser(description="Map Coloring CSP — visualize or batch experiments")
	parser.add_argument('--batch', action='store_true', help='Run batch experiments')
	parser.add_argument('--repeats', type=int, default=5, help='Repeats per configuration in batch mode')
	parser.add_argument('--csv', type=str, default=None, help='Optional CSV output path for per-trial rows')
	parser.add_argument('--randomize-without', action='store_true', help='Randomize variable order for non-heuristic runs in batch mode')
	parser.add_argument('--time-limit-ms', type=int, default=None, help='Optional per-trial time limit (ms) for non-heuristic runs in batch mode')
	parser.add_argument('--backtrack-cap', type=int, default=None, help='Optional per-trial backtrack cap for non-heuristic runs in batch mode')
	parser.add_argument('--all', action='store_true', help='Run USA and Australia for heuristic and non-heuristic, with visualization')
	parser.add_argument('--method', type=int, default=3, choices=[1,2,3], help='Method to use with --all (default: 3)')
	parser.add_argument('--logfile', type=str, default=None, help='Optional log file to append detailed results for --all')
	args = parser.parse_args()

	if args.all:
		run_all(method_choice=args.method, logfile=args.logfile)
	elif args.batch:
		run_batch(repeats=args.repeats, randomize_without=args.randomize_without, output_csv=args.csv,
			  time_limit_ms=args.time_limit_ms, backtrack_cap=args.backtrack_cap)
	else:
		interactive_cli()


if __name__ == "__main__":
	main()

