# Map Coloring using Constraint Satisfaction Problems (CSP)

## Project Overview
This project implements and compares various map coloring algorithms using Constraint Satisfaction Problem (CSP) techniques for USA and Australia maps.

**ITCS 6150 - Intelligent Systems - Project 3**

## Single File Implementation
Everything is contained in one file: **`map_coloring_csp.py`** (~550 lines)

## Usage

### Run All Experiments
```bash
python map_coloring_csp.py
```

This single command will:
1. Run experiments for both Australia and USA maps
2. Test 3 algorithms: DFS, DFS+Forward Checking, DFS+FC+AC-3
3. Run without heuristics (5 runs with random variable ordering)
4. Run with heuristics (5 runs using MRV, Degree, and LCV)
5. Display results in formatted tables
6. Show comparative analysis

**Note:** USA experiments without heuristics may take several minutes.

## Implemented Features

### Algorithms
1. **DFS Only** - Basic backtracking search
2. **DFS + Forward Checking** - Constraint propagation through forward checking
3. **DFS + FC + AC-3** - Arc consistency propagation through singleton domains

### Heuristics
- **MRV (Minimum Remaining Values)** - Choose variable with fewest legal values
- **Degree Constraint** - Choose variable involved in most constraints
- **LCV (Least Constraining Value)** - Choose value that rules out fewest choices

### Maps Included
- **Australia Map**: 7 regions, chromatic number = 3
- **USA Map**: 48 contiguous states, chromatic number = 4

## What the Program Does

### Without Heuristics
Variables are ordered randomly before each run to test raw algorithm performance.

### With Heuristics
Uses intelligent variable and value ordering (MRV, Degree, LCV) for optimal performance.

## Results Format

The program displays tables showing:
- Algorithm name and configuration
- Average backtracks across runs
- Min/Max backtracks
- Average execution time
- Comparative analysis between approaches

## Requirements

- **Python 3.7 or higher**
- **No external dependencies** - Uses only Python standard library

## Project Requirements Met

✅ Compute chromatic number for both maps  
✅ Implement DFS only  
✅ Implement DFS + Forward Checking  
✅ Implement DFS + FC + Propagation through singleton domains (AC-3)  
✅ Run without heuristics with random variable ordering  
✅ Run with heuristics (MRV, Degree, LCV)  
✅ Run each configuration 5+ times  
✅ Track number of backtracks  
✅ Track time required  
✅ Present results in tabular format  
✅ Well-documented code (inline documentation)  

## Expected Performance

### Australia (7 regions, 3 colors)
- With heuristics: All algorithms typically achieve 0 backtracks in ~0.001s

### USA (48 states, 4 colors)
- With heuristics: All algorithms typically achieve 0 backtracks in ~0.02-0.06s
- Without heuristics: Variable performance depending on random ordering

## File Structure
```
ITCS6150-Project3/
├── map_coloring_csp.py    # Complete implementation (run this!)
└── README.md              # This file
```

## Quick Start
```bash
python map_coloring_csp.py
```

## Author
ITCS 6150 Student

## Date
October 30, 2025