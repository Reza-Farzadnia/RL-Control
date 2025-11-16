# CSTR Optimal Control with Pyomo

This is a beginner-friendly Python implementation of a **Continuous Stirred Tank Reactor (CSTR)** optimal control problem using **Pyomo**.

## Problem Description

The optimization problem involves:

### State Variables
- **CR**: Reactant concentration in the reactor [mol/m³]
- **TR**: Reactor temperature [K]
- **TJ**: Jacket (cooling) temperature [K]

### Control Variable
- **FJc**: Jacket coolant flow rate [m³/hr] (controlled by a PI controller)

### Design Variables (optimized)
- **DR**: Reactor diameter [m]
- **HR**: Reactor height [m]
- **Kc**: PI controller gain
- **tau_I**: PI controller integral time [hr]

### Objective
Minimize the initial reactor temperature while maintaining stability.

## Features

- Clean, well-commented code for beginners
- Backward Euler discretization for ODE dynamics
- PI controller implementation
- Beautiful visualization of results
- Saves results to CSV file

## Installation

### 1. Install Python (if not already installed)
Download Python 3.8+ from [python.org](https://www.python.org/)

### 2. Install Required Packages

```bash
pip install -r requirements_pyomo.txt
```

Or install manually:

```bash
pip install pyomo numpy pandas matplotlib
```

### 3. Install IPOPT Solver

**Windows:**
```bash
conda install -c conda-forge ipopt
```

Or download the binary from: [IPOPT releases](https://github.com/coin-or/Ipopt/releases)

**Linux/Mac:**
```bash
conda install -c conda-forge ipopt
```

## Usage

Simply run the Python script:

```bash
python cstr_optimal_control.py
```

## Output

The code generates:

1. **Console output**: Optimal design parameters and solution status
2. **cstr_results.csv**: Detailed results at each time step
3. **cstr_results.png**: Four-panel plot showing:
   - Reactor temperature over time
   - Jacket temperature over time
   - Coolant flow rate (control input)
   - Control error

## Understanding the Code

### Code Structure

```
cstr_optimal_control.py
│
├── Parameters class         # All physical constants and bounds
├── build_cstr_model()      # Pyomo model construction
│   ├── Variables
│   ├── Objective function
│   └── Constraints
├── solve_model()           # Solver interface
├── extract_results()       # Process solution
├── plot_results()          # Visualization
└── main()                  # Execution flow
```

### Key Mathematical Components

1. **Reaction Kinetics** (Arrhenius equation):
   ```
   k1 = k0 * exp(-Ea_R / TR)
   ```

2. **Material Balance** (Backward Euler):
   ```
   V_R * CR[n] = V_R * CR[n-1] + dn * (FR*(CR_in - CR[n]) - k1[n]*V_R*CR[n])
   ```

3. **Energy Balance** (Reactor):
   ```
   V_R * TR[n] = V_R * TR[n-1] + dn * (Heat_in - Heat_reaction - Heat_transfer)
   ```

4. **PI Controller**:
   ```
   dU[k] = Kc * (e[k] - e[k-1] + (dt/tau_I)*e[k])
   FJc[k] = FJc[k-1] + dU[k]
   ```

## Customization

You can easily modify parameters in the `Parameters` class:

```python
class Parameters:
    # Change time horizon
    tf = 2.0  # from 1.0 to 2.0 hours

    # Change discretization
    dn = 0.1  # coarser grid

    # Change bounds
    TR_min, TR_max = 330.0, 365.0
```

## Troubleshooting

### Solver Not Found
If you get "Solver ipopt not found", install IPOPT as described above.

### Slow Convergence
- Increase `max_iter` in solver options
- Try better initial guesses
- Use a coarser time grid (larger `dn` or `dt`)

### Infeasible Solution
- Check bounds on variables
- Relax endpoint tolerance (`tol`)
- Verify physical parameters make sense

## Comparison with Julia Version

This Python/Pyomo code is equivalent to the original Julia/JuMP code with:
- Same mathematical formulation
- Same discretization scheme
- Cleaner structure for readability
- Better comments for learning

## Further Reading

- [Pyomo Documentation](https://pyomo.readthedocs.io/)
- [IPOPT Documentation](https://coin-or.github.io/Ipopt/)
- Chemical Reactor Control textbooks

## License

Educational use - feel free to modify and extend!
