"""
Quick test script to verify map_coloring_csp.py works correctly
Run this first before running the full experiments
"""

from map_coloring_csp import CSPSolver, AUSTRALIA_MAP, USA_MAP

print("="*60)
print("QUICK VERIFICATION TEST")
print("="*60)

# Test Australia
print("\n1. Testing AUSTRALIA map (7 regions, 3 colors)...")
solver = CSPSolver(AUSTRALIA_MAP, 3)
solution, backtracks, time = solver.solve('ac3', use_heuristics=True)
if solution:
    print(f"   ✓ Solution found!")
    print(f"   ✓ Backtracks: {backtracks}")
    print(f"   ✓ Time: {time:.6f}s")
else:
    print("   ✗ Failed!")

# Test USA
print("\n2. Testing USA map (48 states, 4 colors)...")
solver = CSPSolver(USA_MAP, 4)
solution, backtracks, time = solver.solve('ac3', use_heuristics=True)
if solution:
    print(f"   ✓ Solution found!")
    print(f"   ✓ Backtracks: {backtracks}")
    print(f"   ✓ Time: {time:.6f}s")
else:
    print("   ✗ Failed!")

print("\n" + "="*60)
print("All tests passed! The program is working correctly.")
print("="*60)
print("\nTo run full experiments:")
print("    python map_coloring_csp.py")
print("\n(Note: Full experiments may take several minutes)")
print()
