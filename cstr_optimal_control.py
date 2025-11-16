"""
Optimal Control of a CSTR (Continuous Stirred Tank Reactor) with PI Control

This code solves an optimal control problem for a chemical reactor with:
- Reactor temperature (TR) and jacket temperature (TJ) dynamics
- Concentration (CR) dynamics
- PI controller for temperature regulation
- Design optimization (reactor dimensions and controller parameters)

Author: Converted from Julia/JuMP to Python/Pyomo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory

# =============================================================================
# PARAMETERS
# =============================================================================

class Parameters:
    """Container for all model parameters"""

    # Time discretization
    dn = 0.05           # ODE integration time step [hr]
    dt = 0.1            # Control update time step [hr]
    tf = 1.0            # Total time horizon [hr]

    N = int(tf / dn) + 1    # Number of ODE steps (21)
    K = int(tf / dt) + 1    # Number of control steps (11)
    Kinc = int(dt / dn)     # ODE steps per control update (2)

    # Physical parameters - Reaction and heat transfer
    dH = -31652.0       # Heat of reaction [kJ/mol]
    U = 1893.0          # Overall heat transfer coefficient [kJ/(hr·m²·K)]
    Ea_R = 8375.0       # Activation energy / R [K]
    k0 = 4.08e10        # Pre-exponential factor [hr⁻¹]

    # Fluid properties
    rho = 800.9         # Reactor fluid density [kg/m³]
    Cp = 0.9691         # Reactor fluid heat capacity [kJ/(kg·K)]
    rhoj = 997.95       # Jacket fluid density [kg/m³]
    Cj = 1.292          # Jacket fluid heat capacity [kJ/(kg·K)]

    # Operating conditions
    FR = 2.832          # Reactor flow rate [m³/hr]
    TR_in0 = 333.0      # Initial inlet temperature [K]
    deltaTR_in = 0.5    # Temperature disturbance step [K]
    TJ_in = 294.1       # Jacket inlet temperature [K]
    CR_in = 10.89       # Inlet concentration [mol/m³]

    # Variable bounds
    TR_min, TR_max = 333.0, 360.8
    TJ_min, TJ_max = 300.0, 1110.0
    FJc_min, FJc_max = 20.553, 20.6
    DR_min, DR_max = 3.0, 6.0       # Reactor diameter [m]
    HR_min, HR_max = 4.0, 10.0      # Reactor height [m]
    Kc_min, Kc_max = -30.0, 10.0    # Controller gain
    tau_min, tau_max = 0.1, 4.0     # Integral time [hr]

    # Constraints
    tol = 5e-4          # Tolerance for endpoint constraint
    n_tol = 2           # Number of endpoint steps to constrain

    # Initial guesses (from previous solution)
    CR_opt = 0.275
    FJ_opt = 20.554
    TJ_opt = 330.005
    TR_opt = 333.0
    Kc_opt = -30.0
    tau_I_opt = 0.873
    HR_opt = 10.0
    DR_opt = 5.342


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_cstr_model():
    """Build the Pyomo optimization model for the CSTR control problem"""

    model = ConcreteModel(name="CSTR_Optimal_Control")
    p = Parameters()

    # Create time vectors
    model.time_N = [round(i * p.dn, 2) for i in range(p.N)]
    model.time_K = [round(k * p.dt, 2) for k in range(p.K)]

    # -------------------------------------------------------------------------
    # SETS
    # -------------------------------------------------------------------------
    model.N_idx = RangeSet(0, p.N - 1)  # ODE time indices
    model.K_idx = RangeSet(0, p.K - 1)  # Control time indices

    # -------------------------------------------------------------------------
    # VARIABLES
    # -------------------------------------------------------------------------

    # State variables
    model.CR = Var(model.N_idx, domain=NonNegativeReals, initialize=p.CR_opt)
    model.TR = Var(model.N_idx, bounds=(p.TR_min, p.TR_max), initialize=p.TR_opt)
    model.TJ = Var(model.N_idx, bounds=(p.TJ_min, p.TJ_max), initialize=p.TJ_opt)

    # Control variables
    model.FJc = Var(model.K_idx, bounds=(p.FJc_min, p.FJc_max), initialize=p.FJ_opt)

    # Reaction rate
    model.k1 = Var(model.N_idx, initialize=1e4)

    # Controller variables
    model.dU = Var(model.K_idx, initialize=0.0)  # Control increment
    model.e = Var(model.K_idx, initialize=0.0)   # Control error

    # Design variables
    model.DR = Var(bounds=(p.DR_min, p.DR_max), initialize=p.DR_opt)  # Diameter
    model.HR = Var(bounds=(p.HR_min, p.HR_max), initialize=p.HR_opt)  # Height
    model.Kc = Var(bounds=(p.Kc_min, p.Kc_max), initialize=p.Kc_opt)  # Controller gain
    model.tau_I = Var(bounds=(p.tau_min, p.tau_max), initialize=p.tau_I_opt)  # Integral time

    # -------------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -------------------------------------------------------------------------
    # Minimize initial reactor temperature
    model.obj = Objective(expr=model.TR[0], sense=minimize)

    # -------------------------------------------------------------------------
    # CONSTRAINTS
    # -------------------------------------------------------------------------

    # Helper function for reactor volume
    def reactor_volume(model):
        return 0.25 * 3.14159 * model.DR**2 * model.HR

    # Helper function for reactor area
    def reactor_area(model):
        return 3.14159 * model.DR * model.HR

    # Helper function for jacket volume
    def jacket_volume(model):
        return (1.0/3.0) * 3.14159 * model.DR * model.HR

    # Reaction rate constraints (Arrhenius equation)
    def reaction_rate_rule(model, n):
        return model.k1[n] == p.k0 * exp(-p.Ea_R / model.TR[n])
    model.reaction_rate_con = Constraint(model.N_idx, rule=reaction_rate_rule)

    # Initial conditions (algebraic equations at t=0)
    def initial_CR_rule(model):
        V_R = reactor_volume(model)
        return 0 == p.FR * (p.CR_in - model.CR[0]) - model.k1[0] * V_R * model.CR[0]
    model.initial_CR = Constraint(rule=initial_CR_rule)

    def initial_TR_rule(model):
        V_R = reactor_volume(model)
        A_R = reactor_area(model)
        heat_reaction = (p.dH / (p.rho * p.Cp)) * model.k1[0] * V_R * model.CR[0]
        heat_transfer = (p.U / (p.rho * p.Cp)) * A_R * (model.TR[0] - model.TJ[0])
        return 0 == p.FR * (p.TR_in0 - model.TR[0]) - heat_reaction - heat_transfer
    model.initial_TR = Constraint(rule=initial_TR_rule)

    def initial_TJ_rule(model):
        A_R = reactor_area(model)
        heat_transfer = (p.U / (p.rhoj * p.Cj)) * A_R * (model.TR[0] - model.TJ[0])
        return 0 == model.FJc[0] * (p.TJ_in - model.TJ[0]) + heat_transfer
    model.initial_TJ = Constraint(rule=initial_TJ_rule)

    # Dynamic constraints (Backward Euler discretization)
    def dynamics_CR_rule(model, n):
        if n == 0:
            return Constraint.Skip
        V_R = reactor_volume(model)
        rhs = p.FR * (p.CR_in - model.CR[n]) - model.k1[n] * V_R * model.CR[n]
        return V_R * model.CR[n] == V_R * model.CR[n-1] + p.dn * rhs
    model.dynamics_CR = Constraint(model.N_idx, rule=dynamics_CR_rule)

    def dynamics_TR_rule(model, n):
        if n == 0:
            return Constraint.Skip

        # Determine inlet temperature (step change at t=0.1)
        TR_in = p.TR_in0 + p.deltaTR_in if model.time_N[n-1] >= 0.1 else p.TR_in0

        V_R = reactor_volume(model)
        A_R = reactor_area(model)
        heat_reaction = (p.dH / (p.rho * p.Cp)) * model.k1[n] * V_R * model.CR[n]
        heat_transfer = (p.U / (p.rho * p.Cp)) * A_R * (model.TR[n] - model.TJ[n])
        rhs = p.FR * (TR_in - model.TR[n]) - heat_reaction - heat_transfer
        return V_R * model.TR[n] == V_R * model.TR[n-1] + p.dn * rhs
    model.dynamics_TR = Constraint(model.N_idx, rule=dynamics_TR_rule)

    def dynamics_TJ_rule(model, n):
        if n == 0:
            return Constraint.Skip

        # Map ODE step to control index
        k = (n - 1) // p.Kinc

        V_J = jacket_volume(model)
        A_R = reactor_area(model)
        heat_transfer = (p.U / (p.rhoj * p.Cj)) * A_R * (model.TR[n] - model.TJ[n])
        rhs = model.FJc[k] * (p.TJ_in - model.TJ[n]) + heat_transfer
        return V_J * model.TJ[n] == V_J * model.TJ[n-1] + p.dn * rhs
    model.dynamics_TJ = Constraint(model.N_idx, rule=dynamics_TJ_rule)

    # PI Controller constraints
    def control_increment_rule(model, k):
        if k == 0:
            return model.dU[k] == 0
        else:
            return model.dU[k] == model.Kc * (
                model.e[k] - model.e[k-1] + (p.dt / model.tau_I) * model.e[k]
            )
    model.control_increment = Constraint(model.K_idx, rule=control_increment_rule)

    def control_update_rule(model, k):
        if k == 0:
            return Constraint.Skip
        else:
            return model.FJc[k] == model.FJc[k-1] + model.dU[k]
    model.control_update = Constraint(model.K_idx, rule=control_update_rule)

    def control_error_rule(model, k):
        # Map control index to ODE index
        n = min(k * p.Kinc, p.N - 1)
        if k == 0:
            return model.e[k] == 0
        else:
            return model.e[k] == model.TR[0] - model.TR[n]
    model.control_error = Constraint(model.K_idx, rule=control_error_rule)

    # Endpoint tolerance constraint (stabilization)
    def endpoint_tolerance_rule(model, n):
        if n < p.N - p.n_tol:
            return Constraint.Skip
        else:
            deviation = model.TR[n] - model.TR[0]
            return (-p.tol, deviation, p.tol)
    model.endpoint_tolerance = Constraint(model.N_idx, rule=endpoint_tolerance_rule)

    return model, p


# =============================================================================
# SOLVER
# =============================================================================

def solve_model(model, solver_name='ipopt'):
    """
    Solve the optimization model

    Parameters:
    -----------
    model : Pyomo ConcreteModel
        The model to solve
    solver_name : str
        Solver to use ('ipopt', 'baron', etc.)
    """

    # Create solver
    solver = SolverFactory(solver_name)

    # Set solver options for IPOPT
    if solver_name == 'ipopt':
        solver.options['max_iter'] = 3000
        solver.options['tol'] = 1e-6
        solver.options['print_level'] = 5

    # Solve
    print(f"\n{'='*60}")
    print(f"Solving with {solver_name.upper()}...")
    print(f"{'='*60}\n")

    results = solver.solve(model, tee=True)

    # Check solution status
    print(f"\n{'='*60}")
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Solution Status: OPTIMAL")
        print(f"Objective Value: {value(model.obj):.6f}")
    else:
        print(f"Solution Status: {results.solver.termination_condition}")
    print(f"{'='*60}\n")

    return results


# =============================================================================
# RESULTS EXTRACTION
# =============================================================================

def extract_results(model, p):
    """Extract results from solved model into a DataFrame"""

    # Extract values from model
    CR_vals = [value(model.CR[n]) for n in model.N_idx]
    TR_vals = [value(model.TR[n]) for n in model.N_idx]
    TJ_vals = [value(model.TJ[n]) for n in model.N_idx]
    k1_vals = [value(model.k1[n]) for n in model.N_idx]

    FJc_vals = [value(model.FJc[k]) for k in model.K_idx]
    dU_vals = [value(model.dU[k]) for k in model.K_idx]
    e_vals = [value(model.e[k]) for k in model.K_idx]

    # Map control variables to ODE time grid
    FJc_per_n = [FJc_vals[n // p.Kinc] for n in range(p.N)]
    dU_per_n = [dU_vals[n // p.Kinc] for n in range(p.N)]
    e_per_n = [e_vals[n // p.Kinc] for n in range(p.N)]

    # Create DataFrame
    df = pd.DataFrame({
        'time': model.time_N,
        'CR': CR_vals,
        'TR': TR_vals,
        'TJ': TJ_vals,
        'k1': k1_vals,
        'FJc': FJc_per_n,
        'dU': dU_per_n,
        'e': e_per_n
    })

    # Print optimal design parameters
    print("\nOptimal Design Parameters:")
    print(f"{'='*60}")
    print(f"Reactor Diameter (DR):    {value(model.DR):.4f} m")
    print(f"Reactor Height (HR):      {value(model.HR):.4f} m")
    print(f"Controller Gain (Kc):     {value(model.Kc):.4f}")
    print(f"Integral Time (tau_I):    {value(model.tau_I):.4f} hr")
    print(f"{'='*60}")
    print(f"\nInitial Conditions:")
    print(f"{'='*60}")
    print(f"Initial Concentration:    {value(model.CR[0]):.6f} mol/m³")
    print(f"Initial Reactor Temp:     {value(model.TR[0]):.6f} K")
    print(f"Initial Jacket Temp:      {value(model.TJ[0]):.6f} K")
    print(f"Initial Flow Rate:        {value(model.FJc[0]):.6f} m³/hr")
    print(f"{'='*60}\n")

    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(df, save_path='cstr_results.png'):
    """Create publication-quality plots of the results"""

    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('CSTR Optimal Control Results', fontsize=16, fontweight='bold')

    # Plot 1: Reactor Temperature
    ax1 = axes[0, 0]
    ax1.plot(df['time'], df['TR'], 'b-', linewidth=2, label='Reactor Temp (TR)')

    # Mark the peak temperature
    max_idx = df['TR'].idxmax()
    t_max = df.loc[max_idx, 'time']
    TR_max = df.loc[max_idx, 'TR']
    ax1.plot(t_max, TR_max, 'ro', markersize=8, label='Peak')
    ax1.annotate(f'{TR_max:.3f} K',
                 xy=(t_max, TR_max),
                 xytext=(10, -10),
                 textcoords='offset points',
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    ax1.set_ylabel('Reactor Temperature (K)', fontsize=11)
    ax1.set_xlabel('Time (hr)', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Jacket Temperature
    ax2 = axes[0, 1]
    ax2.plot(df['time'], df['TJ'], 'g-', linewidth=2, label='Jacket Temp (TJ)')
    ax2.set_ylabel('Jacket Temperature (K)', fontsize=11)
    ax2.set_xlabel('Time (hr)', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Flow Rate (step plot)
    ax3 = axes[1, 0]
    ax3.step(df['time'], df['FJc'], 'g-', linewidth=2, where='post',
             label='Jacket Flow Rate')
    ax3.set_ylabel('Flow Rate (m³/hr)', fontsize=11)
    ax3.set_xlabel('Time (hr)', fontsize=11)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Control Error
    ax4 = axes[1, 1]
    ax4.step(df['time'], df['e'], 'purple', linewidth=2, where='post',
             label='Control Error')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_ylabel('Control Error (K)', fontsize=11)
    ax4.set_xlabel('Time (hr)', fontsize=11)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*60)
    print("CSTR OPTIMAL CONTROL PROBLEM")
    print("="*60)

    # Build model
    print("\nBuilding model...")
    model, params = build_cstr_model()
    print(f"Model built successfully!")
    print(f"  - Number of ODE steps: {params.N}")
    print(f"  - Number of control steps: {params.K}")
    print(f"  - Time horizon: {params.tf} hr")

    # Solve model
    results = solve_model(model, solver_name='ipopt')

    # Extract and save results
    df = extract_results(model, params)
    df.to_csv('cstr_results.csv', index=False)
    print("Results saved to: cstr_results.csv")

    # Plot results
    plot_results(df, save_path='cstr_results.png')

    print("\n" + "="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
