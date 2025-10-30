"""
Map Coloring using Constraint Satisfaction Problems (CSP)
ITCS 6150 - Intelligent Systems - Project 3

This program implements and compares map coloring algorithms using CSP techniques
for USA and Australia maps.

Author: ITCS 6150 Student
Date: October 30, 2025

USAGE:
    python map_coloring_csp.py

The program will:
1. Run experiments on both Australia and USA maps
2. Test 3 algorithms: DFS, DFS+Forward Checking, DFS+FC+AC-3
3. Run without heuristics (random variable ordering) - 5 runs each
4. Run with heuristics (MRV, Degree, LCV) - 5 runs each
5. Display results in formatted tables
6. Show comparative analysis
"""

import time
import random
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Optional


# ============================================================================
# MAP DATA
# ============================================================================

# USA Map - 48 contiguous states and their neighbors
USA_MAP = {
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
    'ME': ['NH']
}

# Australia Map - States/territories and their neighbors
AUSTRALIA_MAP = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'QLD'],
    'SA': ['WA', 'NT', 'QLD', 'NSW', 'VIC'],
    'QLD': ['NT', 'SA', 'NSW'],
    'NSW': ['QLD', 'SA', 'VIC'],
    'VIC': ['SA', 'NSW'],
    'TAS': []
}


# ============================================================================
# CSP SOLVER
# ============================================================================

class CSPSolver:
    """CSP solver for map coloring with various algorithms and heuristics"""
    
    def __init__(self, adjacency_map: Dict[str, List[str]], num_colors: int):
        self.adjacency_map = adjacency_map
        self.num_colors = num_colors
        self.colors = list(range(num_colors))
        self.variables = list(adjacency_map.keys())
        self.backtrack_count = 0
        self.start_time = 0
        self.end_time = 0
        
    def reset_statistics(self):
        """Reset statistics counters"""
        self.backtrack_count = 0
        self.start_time = 0
        self.end_time = 0
        
    def is_consistent(self, assignment: Dict[str, int], var: str, color: int) -> bool:
        """Check if assigning a color to a variable is consistent"""
        for neighbor in self.adjacency_map[var]:
            if neighbor in assignment and assignment[neighbor] == color:
                return False
        return True
    
    def select_unassigned_variable(self, assignment: Dict[str, int], 
                                   domains: Dict[str, Set[int]],
                                   use_mrv: bool = False,
                                   use_degree: bool = False) -> Optional[str]:
        """Select next unassigned variable using heuristics"""
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None
            
        if not use_mrv:
            return unassigned[0]
        
        # MRV: Choose variable with smallest domain
        min_remaining = min(len(domains[v]) for v in unassigned)
        candidates = [v for v in unassigned if len(domains[v]) == min_remaining]
        
        if not use_degree or len(candidates) == 1:
            return candidates[0]
        
        # Degree heuristic: Choose variable with most constraints
        def count_unassigned_neighbors(var):
            return sum(1 for neighbor in self.adjacency_map[var] if neighbor not in assignment)
        
        return max(candidates, key=count_unassigned_neighbors)
    
    def order_domain_values(self, var: str, assignment: Dict[str, int],
                           domains: Dict[str, Set[int]],
                           use_lcv: bool = False) -> List[int]:
        """Order domain values using Least Constraining Value heuristic"""
        if not use_lcv:
            return list(domains[var])
        
        # LCV: Prefer values that rule out fewest choices for neighbors
        def count_conflicts(color):
            conflicts = 0
            for neighbor in self.adjacency_map[var]:
                if neighbor not in assignment and color in domains[neighbor]:
                    conflicts += 1
            return conflicts
        
        return sorted(domains[var], key=count_conflicts)
    
    def forward_check(self, var: str, color: int, assignment: Dict[str, int],
                     domains: Dict[str, Set[int]]) -> Optional[Dict[str, Set[int]]]:
        """Perform forward checking"""
        new_domains = deepcopy(domains)
        
        for neighbor in self.adjacency_map[var]:
            if neighbor not in assignment:
                if color in new_domains[neighbor]:
                    new_domains[neighbor].remove(color)
                
                if not new_domains[neighbor]:
                    return None
        
        return new_domains
    
    def propagate_constraints(self, domains: Dict[str, Set[int]], 
                            assignment: Dict[str, int]) -> Optional[Dict[str, Set[int]]]:
        """Propagate constraints through singleton domains (AC-3)"""
        new_domains = deepcopy(domains)
        queue = []
        
        for var in self.variables:
            if var not in assignment:
                for neighbor in self.adjacency_map[var]:
                    if neighbor not in assignment:
                        queue.append((var, neighbor))
        
        while queue:
            (xi, xj) = queue.pop(0)
            
            if xi in assignment or xj in assignment:
                continue
                
            if self.revise(new_domains, xi, xj):
                if not new_domains[xi]:
                    return None
                
                for neighbor in self.adjacency_map[xi]:
                    if neighbor != xj and neighbor not in assignment:
                        queue.append((neighbor, xi))
        
        return new_domains
    
    def revise(self, domains: Dict[str, Set[int]], xi: str, xj: str) -> bool:
        """Revise domain of xi based on xj (AC-3)"""
        revised = False
        to_remove = set()
        
        for color_i in domains[xi]:
            satisfies = any(color_i != color_j for color_j in domains[xj])
            if not satisfies:
                to_remove.add(color_i)
                revised = True
        
        domains[xi] -= to_remove
        return revised
    
    def backtrack(self, assignment: Dict[str, int], domains: Dict[str, Set[int]],
                 forward_checking: bool = False, 
                 constraint_propagation: bool = False,
                 use_mrv: bool = False,
                 use_degree: bool = False,
                 use_lcv: bool = False) -> Optional[Dict[str, int]]:
        """Recursive backtracking search"""
        if len(assignment) == len(self.variables):
            return assignment
        
        var = self.select_unassigned_variable(assignment, domains, use_mrv, use_degree)
        if var is None:
            return assignment
        
        ordered_values = self.order_domain_values(var, assignment, domains, use_lcv)
        
        for color in ordered_values:
            if self.is_consistent(assignment, var, color):
                assignment[var] = color
                new_domains = deepcopy(domains)
                new_domains[var] = {color}
                
                inferences_valid = True
                
                if forward_checking:
                    new_domains = self.forward_check(var, color, assignment, new_domains)
                    if new_domains is None:
                        inferences_valid = False
                
                if inferences_valid and constraint_propagation:
                    new_domains = self.propagate_constraints(new_domains, assignment)
                    if new_domains is None:
                        inferences_valid = False
                
                if inferences_valid:
                    result = self.backtrack(assignment, new_domains, forward_checking,
                                          constraint_propagation, use_mrv, use_degree, use_lcv)
                    if result is not None:
                        return result
                
                del assignment[var]
                self.backtrack_count += 1
        
        return None
    
    def solve(self, algorithm: str = 'backtrack', use_heuristics: bool = False,
             variable_order: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, int]], int, float]:
        """Solve the CSP using specified algorithm"""
        self.reset_statistics()
        
        if variable_order:
            self.variables = variable_order
        
        domains = {var: set(self.colors) for var in self.variables}
        
        forward_checking = algorithm in ['forward_checking', 'ac3']
        constraint_propagation = algorithm == 'ac3'
        
        self.start_time = time.time()
        
        assignment = {}
        solution = self.backtrack(
            assignment, 
            domains,
            forward_checking=forward_checking,
            constraint_propagation=constraint_propagation,
            use_mrv=use_heuristics,
            use_degree=use_heuristics,
            use_lcv=use_heuristics
        )
        
        self.end_time = time.time()
        time_elapsed = self.end_time - self.start_time
        
        return solution, self.backtrack_count, time_elapsed


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiments(map_name: str, adjacency_map: Dict, num_colors: int, num_runs: int = 5):
    """Run all experiments for a given map"""
    
    algorithms = [
        ('backtrack', 'DFS Only'),
        ('forward_checking', 'DFS + Forward Checking'),
        ('ac3', 'DFS + FC + AC-3')
    ]
    
    print(f"\n{'='*80}")
    print(f"MAP: {map_name}")
    print(f"Regions: {len(adjacency_map)} | Colors: {num_colors}")
    print(f"{'='*80}\n")
    
    # Run WITHOUT heuristics
    print(f"{'='*80}")
    print("EXPERIMENTS WITHOUT HEURISTICS (Random Variable Ordering)")
    print(f"{'='*80}\n")
    
    results_without = {}
    variables = list(adjacency_map.keys())
    
    for algo_key, algo_name in algorithms:
        results_without[algo_name] = []
        
        for run in range(num_runs):
            random_order = variables.copy()
            random.shuffle(random_order)
            
            print(f"Running {algo_name} (without heuristics) - Run {run + 1}/{num_runs}...")
            
            solver = CSPSolver(adjacency_map, num_colors)
            solution, backtracks, elapsed = solver.solve(
                algorithm=algo_key,
                use_heuristics=False,
                variable_order=random_order
            )
            
            results_without[algo_name].append({
                'backtracks': backtracks,
                'time': elapsed,
                'success': solution is not None
            })
            
            if solution:
                print(f"  ✓ Success - Backtracks: {backtracks}, Time: {elapsed:.6f}s")
            else:
                print(f"  ✗ Failed")
    
    # Run WITH heuristics
    print(f"\n{'='*80}")
    print("EXPERIMENTS WITH HEURISTICS (MRV, Degree, LCV)")
    print(f"{'='*80}\n")
    
    results_with = {}
    
    for algo_key, algo_name in algorithms:
        results_with[algo_name] = []
        
        for run in range(num_runs):
            print(f"Running {algo_name} (with heuristics) - Run {run + 1}/{num_runs}...")
            
            solver = CSPSolver(adjacency_map, num_colors)
            solution, backtracks, elapsed = solver.solve(
                algorithm=algo_key,
                use_heuristics=True
            )
            
            results_with[algo_name].append({
                'backtracks': backtracks,
                'time': elapsed,
                'success': solution is not None
            })
            
            if solution:
                print(f"  ✓ Success - Backtracks: {backtracks}, Time: {elapsed:.6f}s")
            else:
                print(f"  ✗ Failed")
    
    # Calculate and display statistics
    print_results_table(map_name, results_without, "WITHOUT HEURISTICS")
    print_results_table(map_name, results_with, "WITH HEURISTICS")
    
    # Comparison
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS - {map_name}")
    print(f"{'='*80}\n")
    
    for algo_name in results_without.keys():
        without_runs = [r for r in results_without[algo_name] if r['success']]
        with_runs = [r for r in results_with[algo_name] if r['success']]
        
        if without_runs and with_runs:
            without_avg = sum(r['backtracks'] for r in without_runs) / len(without_runs)
            with_avg = sum(r['backtracks'] for r in with_runs) / len(with_runs)
            
            print(f"{algo_name}:")
            print(f"  Without heuristics: {without_avg:.2f} backtracks")
            print(f"  With heuristics:    {with_avg:.2f} backtracks")
            
            if without_avg > 0:
                improvement = ((without_avg - with_avg) / without_avg) * 100
                print(f"  Improvement:        {improvement:.2f}%")
            elif with_avg == 0:
                print(f"  Improvement:        Both optimal (0 backtracks)")
            print()


def print_results_table(map_name: str, results: Dict, experiment_type: str):
    """Print results in formatted table"""
    print(f"\n{'='*100}")
    print(f"RESULTS: {map_name} - {experiment_type}")
    print(f"{'='*100}")
    
    print(f"{'Algorithm':<35} | {'Avg Backtracks':<15} | {'Min/Max Backtracks':<20} | {'Avg Time (s)':<15}")
    print(f"{'-'*35}-+-{'-'*15}-+-{'-'*20}-+-{'-'*15}")
    
    for algo_name, runs in results.items():
        successful = [r for r in runs if r['success']]
        
        if successful:
            backtracks = [r['backtracks'] for r in successful]
            times = [r['time'] for r in successful]
            
            avg_bt = sum(backtracks) / len(backtracks)
            min_bt = min(backtracks)
            max_bt = max(backtracks)
            avg_time = sum(times) / len(times)
            
            print(f"{algo_name:<35} | {avg_bt:<15.2f} | {min_bt}/{max_bt:<18} | {avg_time:<15.6f}")
        else:
            print(f"{algo_name:<35} | {'N/A':<15} | {'N/A':<20} | {'N/A':<15}")
    
    print(f"{'='*100}\n")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main program entry point"""
    print("\n" + "="*80)
    print("CONSTRAINT SATISFACTION PROBLEM (CSP) - MAP COLORING")
    print("ITCS 6150 - Intelligent Systems - Project 3")
    print("="*80)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Number of runs per algorithm
    num_runs = 5
    
    # Run experiments for Australia (fast)
    print("\n" + "#"*80)
    print("# AUSTRALIA MAP EXPERIMENTS")
    print("#"*80)
    run_experiments("AUSTRALIA", AUSTRALIA_MAP, num_colors=3, num_runs=num_runs)
    
    # Run experiments for USA (slower without heuristics)
    print("\n" + "#"*80)
    print("# USA MAP EXPERIMENTS")
    print("#"*80)
    print("\nNOTE: USA experiments without heuristics may take several minutes...")
    run_experiments("USA", USA_MAP, num_colors=4, num_runs=num_runs)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
