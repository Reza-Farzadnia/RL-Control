"""
Tank Optimal Control with Nonlinear Outflow and RTO

Two-layer architecture:
- RTO layer: Computes economic setpoint h_sp (minimize steady-state flow)
- Optimal control: MPC-style trajectory optimization with Pyomo + IPOPT
- Nonlinear outflow: F_out = k_v * sqrt(h)
- Objective: Economic cost + tracking + move suppression
- Backward Euler discretization

Author: Auto-generated
Date: 2025-11-20
"""

import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *

# =============================================================================
# PARAMETERS
# =============================================================================

class Params:
    """System parameters"""
    # Tank
    A = 1.0              # Cross-sectional area [m²]
    k_v = 0.15           # Valve constant [m^(1/2)/s]

    # Limits
    h_min = 1.0          # Min level [m]
    h_safety = 0.5       # Safety constraint [m]
    F_in_min = 0.0       # Min input [m³/s]
    F_in_max = 1.0       # Max input [m³/s]
    delta_F_max = 0.5    # Max input rate [m³/s per step]

    # Time
    dt = 1.0             # Time step [s]
    N = 30               # Time horizon

    # Initial condition
    h_0 = 3.0            # Initial level [m]

    # Cost weights
    c_flow = 1.0         # Flow cost [$/m³]
    Q = 50.0             # Tracking weight [($/m³)/m²]
    S = 0.1              # Move suppression weight [($/m³)/(m³/s)²]


# =============================================================================
# RTO LAYER
# =============================================================================

def solve_rto(params):
    """
    RTO: Minimize steady-state flow cost

    At steady-state: F_in_ss = F_out_ss = k_v * sqrt(h_ss)
    Objective: min F_in_ss = min k_v * sqrt(h_ss)
    Constraint: h_min <= h_ss <= h_max

    Solution: h_sp = h_min (minimize flow by operating at minimum level)
    """
    h_sp = params.h_min
    F_in_ss = params.k_v * np.sqrt(h_sp)
    return h_sp, F_in_ss


# =============================================================================
# OPTIMAL CONTROL (MPC-style)
# =============================================================================

def create_model(params, h_sp, F_in_ss):
    """Create Pyomo optimization model with RTO setpoint"""

    m = ConcreteModel()

    # Time indices
    m.K = RangeSet(0, params.N)

    # Variables
    m.h = Var(m.K, bounds=(params.h_safety, params.h_max))
    m.F_in = Var(m.K, bounds=(params.F_in_min, params.F_in_max))
    m.F_out = Var(m.K, bounds=(0, 2.0))

    # Objective: economic cost + MPC tracking + move suppression
    def obj_rule(m):
        # Economic cost: minimize total flow
        flow_cost = sum(params.c_flow * m.F_in[k] for k in range(1, params.N+1))

        # Tracking cost: penalize deviation from RTO setpoint
        tracking_cost = sum(params.Q * (m.h[k] - h_sp)**2 for k in range(1, params.N+1))

        # Move suppression: penalize input rate of change
        move_cost = sum(params.S * (m.F_in[k] - m.F_in[k-1])**2 for k in range(1, params.N+1))

        return flow_cost + tracking_cost + move_cost
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # Initial condition
    m.init_h = Constraint(expr=m.h[0] == params.h_0)
    m.init_F = Constraint(expr=m.F_in[0] == params.k_v * sqrt(params.h_0))

    # Dynamics (Backward Euler): h[k+1] = h[k] + dt/A * (F_in[k+1] - F_out[k+1])
    def dynamics_rule(m, k):
        if k == 0:
            return Constraint.Skip
        return m.h[k] == m.h[k-1] + (params.dt / params.A) * (m.F_in[k] - m.F_out[k])
    m.dynamics = Constraint(m.K, rule=dynamics_rule)

    # Nonlinear outflow: F_out = k_v * sqrt(h)
    def outflow_rule(m, k):
        return m.F_out[k] == params.k_v * sqrt(m.h[k])
    m.outflow = Constraint(m.K, rule=outflow_rule)

    # Rate limits: |F_in[k] - F_in[k-1]| <= delta_F_max
    def rate_upper_rule(m, k):
        if k == 0:
            return Constraint.Skip
        return m.F_in[k] - m.F_in[k-1] <= params.delta_F_max
    m.rate_upper = Constraint(m.K, rule=rate_upper_rule)

    def rate_lower_rule(m, k):
        if k == 0:
            return Constraint.Skip
        return m.F_in[k] - m.F_in[k-1] >= -params.delta_F_max
    m.rate_lower = Constraint(m.K, rule=rate_lower_rule)

    return m


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":

    params = Params()

    print("=" * 70)
    print("TANK RTO + OPTIMAL CONTROL: NONLINEAR OUTFLOW (PYOMO + IPOPT)")
    print("=" * 70)
    print()
    print(f"System:")
    print(f"  Nonlinear outflow:  F_out = {params.k_v} * sqrt(h)")
    print(f"  Tank area:          A = {params.A} m²")
    print(f"  Time horizon:       N = {params.N} steps")
    print(f"  Time step:          dt = {params.dt} s")
    print()
    print(f"Limits:")
    print(f"  Level:              [{params.h_safety}, {params.h_max}] m")
    print(f"  Input:              [{params.F_in_min}, {params.F_in_max}] m³/s")
    print(f"  Rate limit:         +/- {params.delta_F_max} m³/s")
    print()
    print(f"Costs:")
    print(f"  Flow cost:          c = {params.c_flow} $/m³")
    print(f"  Tracking weight:    Q = {params.Q}")
    print(f"  Move suppression:   S = {params.S}")
    print()

    # RTO layer: compute economic setpoint
    print("RTO Layer: Computing economic setpoint...")
    h_sp, F_in_ss = solve_rto(params)
    print(f"  Setpoint:           h_sp = {h_sp:.3f} m")
    print(f"  Steady-state flow:  F_in_ss = {F_in_ss:.4f} m³/s")
    print()
    print(f"Initial condition:")
    print(f"  Initial level:      h_0 = {params.h_0} m")
    print()

    # Create and solve optimal control problem
    print("Building optimal control model...")
    model = create_model(params, h_sp, F_in_ss)

    print("Solving with IPOPT...")
    solver = SolverFactory('ipopt')
    solver.options['print_level'] = 5
    solver.options['max_iter'] = 500
    results = solver.solve(model, tee=True)

    # Check solution
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("\n" + "=" * 70)
        print("SOLUTION FOUND")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"WARNING: Solver status = {results.solver.termination_condition}")
        print("=" * 70)

    # Extract results
    time = np.array([k * params.dt for k in range(params.N + 1)])
    h = np.array([value(model.h[k]) for k in range(params.N + 1)])
    F_in = np.array([value(model.F_in[k]) for k in range(params.N + 1)])
    F_out = np.array([value(model.F_out[k]) for k in range(params.N + 1)])

    # Statistics
    delta_F = np.diff(F_in)
    max_rate = np.max(np.abs(delta_F))
    violations = np.sum(h < params.h_safety)
    total_cost = value(model.obj)

    print()
    print(f"Results:")
    print(f"  RTO setpoint:       h_sp = {h_sp:.3f} m")
    print(f"  Final level:        h_final = {h[-1]:.3f} m")
    print(f"  Final error:        |h_final - h_sp| = {abs(h[-1] - h_sp):.4f} m")
    print(f"  Min level:          h_min = {np.min(h):.3f} m")
    print(f"  Safety violations:  {violations} steps")
    print(f"  Max input rate:     {max_rate:.4f} m³/s")
    print(f"  Total cost:         {total_cost:.2f}")
    print()

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Tank Level
    axes[0].plot(time, h, 'b-', linewidth=2, label='Level h(t)', marker='o', markersize=4)
    axes[0].axhline(h_sp, color='g', linestyle='--', linewidth=1.5,
                   label=f'RTO Setpoint = {h_sp:.2f} m')
    axes[0].axhline(params.h_safety, color='r', linestyle='-', linewidth=2,
                   label=f'Safety = {params.h_safety:.2f} m')
    axes[0].fill_between(time, 0, params.h_safety, alpha=0.2, color='red', label='Forbidden')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Level [m]')
    axes[0].set_title(f'Tank Level (Min: {np.min(h):.3f} m)')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Flows
    axes[1].step(time, F_in, 'g-', linewidth=2, where='post', label='Input F_in(t)', marker='s', markersize=4)
    axes[1].plot(time, F_out, 'm-', linewidth=2, label='Output F_out(t) = k_v*sqrt(h)', marker='o', markersize=4)
    axes[1].axhline(F_in_ss, color='r', linestyle='--', linewidth=1.5,
                   label=f'RTO steady-state = {F_in_ss:.4f} m³/s')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Flow rate [m³/s]')
    axes[1].set_title('Input and Output Flows (Nonlinear Outflow)')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Input Rate
    axes[2].plot(time[1:], delta_F, 'r-', linewidth=2, label='ΔF_in', marker='o', markersize=4)
    axes[2].axhline(params.delta_F_max, color='k', linestyle='--', linewidth=1.5,
                   label=f'+{params.delta_F_max} m³/s')
    axes[2].axhline(-params.delta_F_max, color='k', linestyle='--', linewidth=1.5,
                   label=f'-{params.delta_F_max} m³/s')
    axes[2].axhline(0, color='gray', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Input rate [m³/s per step]')
    axes[2].set_title(f'Input Rate of Change (Max: {max_rate:.4f} m³/s)')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
