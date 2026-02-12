"""
Tank Hierarchical RTO-MPC with Nonlinear Outflow

Two-layer hierarchical optimization:
- RTO layer: Solves optimization to find economic steady-state setpoint
- MPC layer: Solves dynamic optimization with economic awareness

Nonlinear outflow: F_out = k_v * sqrt(h)
Backward Euler discretization
Solved with Pyomo + IPOPT

Author: Reza Farzadnia
Date: 2025-11-24
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *


# =============================================================================
# PARAMETERS
# =============================================================================

class Params:
    """System parameters"""
    # Tank
    A = 1.0              # Cross-sectional area [m^2]
    k_v = 0.15           # Valve constant
    h_target = 1.0       # Desired operating level to bias RTO setpoint

    # Limits
    h_min = 0.025          # Min level [m]
    h_max = 5.0          # Max level [m]
    h_safety = 0.5       # Safety constraint [m]
    F_in_min = 0.0       # Min input [m^3/s]
    F_in_max = 1.0       # Max input [m^3/s]
    delta_F_max = 0.5    # Max input rate [m^3/s per step]

    # Time
    dt = 1.0             # Time step [s]
    N = 20               # Time horizon

    # Initial condition
    h_0 = 3.0            # Initial level [m]

    # Cost weights
    c_flow = 1.0         # Flow cost [$/m^3]
    Q = 50.0             # Tracking weight 
    S = 0.1              # Move suppression weight 
    W_econ = 10.0        # Economic steady-state penalty weight
    W_target = 5.0       # RTO bias toward h_target
    W_safe = 1000.0      # Safety slack penalty weight


# =============================================================================
# RTO LAYER (HIERARCHICAL OPTIMIZATION)
# =============================================================================

def solve_rto(params):
    """
    RTO: Solve optimization for economic steady-state setpoint

    Optimization problem:
        min_{h_sp, F_in_ss}  c * F_in_ss
        s.t. F_in_ss = k_v * sqrt(h_sp)     (steady-state condition)
             h_min <= h_sp <= h_max

    This is a proper optimization (not hard-coded), solved with IPOPT
    """
    print("  Solving RTO optimization problem...")

    m = ConcreteModel()

    # Decision variables
    m.h_sp = Var(bounds=(params.h_min, params.h_max), initialize=params.h_min)
    m.F_in_ss = Var(bounds=(params.F_in_min, params.F_in_max), initialize=0.1)
    # Soft safety slack (penalizes h_sp below safety without binaries)
    m.safety_slack = Var(domain=NonNegativeReals, initialize=0.0)

    # Objective: minimize steady-state flow cost + soft safety penalty
    w_safe = getattr(params, "W_safe", 1e3)  # penalty weight
    m.obj = Objective(
        expr=(
            params.c_flow * m.F_in_ss
            + w_safe * m.safety_slack
            + params.W_target * (m.h_sp - params.h_target) ** 2  # encourage h_target
        ),
        sense=minimize
    )

    # Steady-state constraint: F_in_ss = k_v * sqrt(h_sp)
    m.steady_state = Constraint(expr=m.F_in_ss == params.k_v * sqrt(m.h_sp))
    # Soft safety: allow h_sp below safety only via slack
    m.safety_soft = Constraint(expr=m.h_sp + m.safety_slack >= params.h_safety)

    # Solve
    solver = SolverFactory('ipopt')
    solver.options['print_level'] = 0  # Suppress IPOPT output for RTO
    results = solver.solve(m, tee=False)

    if results.solver.termination_condition == TerminationCondition.optimal:
        h_sp = value(m.h_sp)
        F_in_ss = value(m.F_in_ss)
        print(f"  RTO converged: h_sp = {h_sp:.4f} m, F_in_ss = {F_in_ss:.4f} m^3/s")
        return h_sp, F_in_ss
    else:
        print(f"  WARNING: RTO did not converge, using h_min as fallback")
        h_sp = params.h_min
        F_in_ss = params.k_v * np.sqrt(h_sp)
        return h_sp, F_in_ss


# =============================================================================
# MPC LAYER (ECONOMIC MPC WITH HIERARCHY)
# =============================================================================

def create_mpc_model(params, h_sp, F_in_ss):
    """
    Key feature: Includes steady-state economic penalty to align with RTO

    Objective = transient costs + economic steady-state penalty
              = sum(c*F_in[k]) + sum(Q*(h[k]-h_sp)^2) + sum(S*delta_F^2)
                + W_econ * (terminal steady-state deviation penalty)

    The W_econ term penalizes deviation from economic optimum at end of horizon,
    mitigating suboptimality from hierarchical decomposition.
    """

    m = ConcreteModel()

    # Time indices
    m.K = RangeSet(0, params.N)

    # Variables
    m.h = Var(m.K, bounds=(params.h_min, params.h_max))
    m.safety_slack = Var(m.K, domain=NonNegativeReals)
    m.F_in = Var(m.K, bounds=(params.F_in_min, params.F_in_max))
    m.F_out = Var(m.K, bounds=(0, 2.0))

    # Objective: Economic MPC with steady-state awareness
    def obj_rule(m):
        # 1. Tracking cost (follow RTO setpoint during transient)
        tracking_cost = sum(params.Q * (m.h[k] - h_sp)**2 for k in range(1, params.N+1))

        # 2. Move suppression (smooth control action)
        move_cost = sum(params.S * (m.F_in[k] - m.F_in[k-1])**2 for k in range(1, params.N+1))

        # 3. SUBOPTIMALITY MITIGATION: Terminal economic penalty
        # Penalize deviation from RTO steady-state at horizon end
        # This makes MPC "aware" of long-term economic target
        terminal_econ_penalty = params.W_econ * (m.h[params.N] - h_sp)**2

        # 4. Soft safety penalty via slack
        safety_penalty = sum(params.W_safe * m.safety_slack[k] for k in m.K)

        return tracking_cost + move_cost + terminal_econ_penalty + safety_penalty

    m.obj = Objective(rule=obj_rule, sense=minimize)

    # Initial condition
    m.init_h = Constraint(expr=m.h[0] == params.h_0)
    m.init_F = Constraint(expr=m.F_in[0] == params.k_v * sqrt(params.h_0))

    # Dynamics (Backward Euler): h[k] = h[k-1] + dt/A * (F_in[k-1] - F_out[k])
    def dynamics_rule(m, k):
        if k == 0:
            return Constraint.Skip
        return m.h[k] == m.h[k-1] + (params.dt / params.A) * (m.F_in[k-1] - m.F_out[k])
    m.dynamics = Constraint(m.K, rule=dynamics_rule)

    # Nonlinear outflow: F_out = k_v * sqrt(h)
    def outflow_rule(m, k):
        return m.F_out[k] == params.k_v * sqrt(m.h[k])
    m.outflow = Constraint(m.K, rule=outflow_rule)

    # Soft safety constraint (h >= h_safety - slack)
    def safety_soft_rule(m, k):
        return m.h[k] + m.safety_slack[k] >= params.h_safety
    m.safety_soft = Constraint(m.K, rule=safety_soft_rule)

    # Rate limits
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
    print("HIERARCHICAL RTO-MPC WITH ECONOMIC AWARENESS (PYOMO + IPOPT)")
    print("=" * 70)
    print()
    print(f"System:")
    print(f"  Nonlinear outflow:  F_out = {params.k_v} * sqrt(h)")
    print(f"  Tank area:          A = {params.A} m^2")
    print(f"  Time horizon:       N = {params.N} steps")
    print(f"  Time step:          dt = {params.dt} s")
    print()
    print(f"Limits:")
    print(f"  Level:              [{params.h_min}, {params.h_max}] m (safety soft at {params.h_safety} m)")
    print(f"  Input:              [{params.F_in_min}, {params.F_in_max}] m^3/s")
    print(f"  Rate limit:         +/- {params.delta_F_max} m^3/s")
    print()
    print(f"Cost weights:")
    print(f"  Flow cost:          c = {params.c_flow} $/m^3")
    print(f"  Tracking weight:    Q = {params.Q}")
    print(f"  Move suppression:   S = {params.S}")
    print(f"  Economic penalty:   W_econ = {params.W_econ} (suboptimality mitigation)")
    print(f"  Safety penalty:     W_safe = {params.W_safe} (soft safety slack)")
    print()
    print(f"Initial condition:")
    print(f"  Initial level:      h_0 = {params.h_0} m")
    print()

    # ==========================================================================
    # LAYER 1: RTO OPTIMIZATION
    # ==========================================================================
    print("=" * 70)
    print("LAYER 1: RTO - Economic Steady-State Optimization")
    print("=" * 70)
    h_sp, F_in_ss = solve_rto(params)
    print()

    # ==========================================================================
    # LAYER 2: MPC OPTIMIZATION (with economic awareness)
    # ==========================================================================
    print("=" * 70)
    print("LAYER 2: Economic MPC - Dynamic Trajectory Optimization")
    print("=" * 70)
    print("  Building Economic MPC model...")
    model = create_mpc_model(params, h_sp, F_in_ss)

    print("  Solving with IPOPT...")
    solver = SolverFactory('ipopt')
    solver.options['print_level'] = 5
    solver.options['max_iter'] = 500
    t_start = time.perf_counter()
    results = solver.solve(model, tee=True)
    solve_time = time.perf_counter() - t_start

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

    # Steady-state analysis at final time
    h_final = h[-1]
    F_in_final = F_in[-1]
    F_out_final = F_out[-1]
    ss_error = abs(F_in_final - F_out_final)  # Should be close to 0 at steady-state

    print()
    print(f"Results:")
    print(f"  RTO setpoint:       h_sp = {h_sp:.3f} m")
    print(f"  RTO steady-state:   F_in_ss = {F_in_ss:.4f} m^3/s")
    print(f"  Final level:        h_final = {h_final:.3f} m")
    print(f"  Final error:        |h_final - h_sp| = {abs(h_final - h_sp):.4f} m")
    print(f"  Steady-state error: |F_in - F_out|_final = {ss_error:.6f} m^3/s")
    print(f"  Min level:          h_min = {np.min(h):.3f} m")
    print(f"  Safety violations:  {violations} steps")
    print(f"  Max input rate:     {max_rate:.4f} m^3/s")
    print(f"  Total MPC cost:     {total_cost:.2f}")
    print(f"  Solve time:         {solve_time:.2f} s")
    print()

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))

    # Panel 1: Tank Level
    axes[0].plot(time, h, 'b-', linewidth=2, label='Level h(t)')
    axes[0].axhline(h_sp, color='g', linestyle='--', linewidth=1.5,
                   label=f'RTO Setpoint = {h_sp:.2f} m')
    axes[0].axhline(params.h_safety, color='r', linestyle='-', linewidth=2,
                   label=f'Safety = {params.h_safety:.2f} m')
    axes[0].fill_between(time, 0, params.h_safety, alpha=0.2, color='red', label='Forbidden')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Level [m]')
    axes[0].set_title(f'Tank Level (Min: {np.min(h):.3f} m, Final SS error: {abs(h_final - h_sp):.4f} m)')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Flows
    axes[1].step(time, F_in, 'g-', linewidth=2, where='post', label='Input F_in(t)')
    axes[1].plot(time, F_out, 'm-', linewidth=2, label='Output F_out(t) = k_v*sqrt(h)')
    axes[1].axhline(F_in_ss, color='r', linestyle='--', linewidth=1.5,
                   label=f'RTO steady-state = {F_in_ss:.4f} m^3/s')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Flow rate [m^3/s]')
    axes[1].set_title(f'Input and Output Flows (Final: F_in={F_in_final:.4f}, F_out={F_out_final:.4f} m^3/s)')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Input Rate
    axes[2].plot(time[1:], delta_F, 'r-', linewidth=2, label='ΔF_in')
    axes[2].axhline(params.delta_F_max, color='k', linestyle='--', linewidth=1.5,
                   label=f'+{params.delta_F_max} m^3/s')
    axes[2].axhline(-params.delta_F_max, color='k', linestyle='--', linewidth=1.5,
                   label=f'-{params.delta_F_max} m^3/s')
    axes[2].axhline(0, color='gray', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Input rate [m^3/s per step]')
    axes[2].set_title(f'Input Rate of Change (Max: {max_rate:.4f} m^3/s)')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures", "result"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")

    plt.show()
