# Quick Start Guide - CSTR Optimal Control

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install pyomo numpy pandas matplotlib seaborn
conda install -c conda-forge ipopt
```

### Step 2: Run the Code

**Windows:**
```cmd
run_cstr.bat
```

**Mac/Linux:**
```bash
python cstr_optimal_control.py
```

**Jupyter Notebook:**
```bash
jupyter notebook cstr_tutorial.ipynb
```

### Step 3: View Results
- Check console for optimal parameters
- Open `cstr_results.png` for plots
- Open `cstr_results.csv` for data

---

## 📊 What You'll Get

### Console Output
```
============================================================
CSTR OPTIMAL CONTROL PROBLEM
============================================================

Optimal Design Parameters:
============================================================
Reactor Diameter (DR):    5.3416 m
Reactor Height (HR):      10.0000 m
Controller Gain (Kc):     -30.0000
Integral Time (tau_I):    0.8732 hr
============================================================
```

### Generated Files
- `cstr_results.csv` - Time-series data
- `cstr_results.png` - Four-panel visualization
- Console log with detailed solver output

---

## 🎯 Problem Overview

**System**: Continuous Stirred Tank Reactor (CSTR) with PI control

**What We're Optimizing**:
- Reactor dimensions (diameter, height)
- PI controller parameters (gain, integral time)
- Control trajectory (jacket flow rate)

**Subject To**:
- Mass and energy balance equations
- PI controller dynamics
- Temperature bounds
- Endpoint stability

**Objective**: Minimize initial reactor temperature

---

## ⚙️ Quick Customization

### Change Time Horizon
```python
# In cstr_optimal_control.py, Parameters class
tf = 2.0  # from 1.0 to 2.0 hours
```

### Change Objective
```python
# In build_cstr_model() function
model.obj = Objective(expr=model.FJc[0], sense=minimize)  # Minimize flow
```

### Relax Tolerance
```python
# In Parameters class
tol = 1e-3  # from 5e-4
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: pyomo` | `pip install pyomo` |
| `Solver ipopt not found` | `conda install -c conda-forge ipopt` |
| Slow convergence | Increase `solver.options['max_iter'] = 5000` |
| Infeasible solution | Relax bounds or tolerances |

---

## 📚 Files Description

| File | Purpose |
|------|---------|
| `cstr_optimal_control.py` | Main Python script |
| `cstr_tutorial.ipynb` | Interactive Jupyter notebook |
| `requirements_pyomo.txt` | Python dependencies |
| `run_cstr.bat` | Windows quick-start script |
| `README_PYOMO.md` | Detailed documentation |
| `PYOMO_CONVERSION_SUMMARY.md` | Complete conversion guide |

---

## 🎓 Learning Path

1. **Beginner**: Run the code, examine outputs
2. **Intermediate**: Modify parameters, try experiments
3. **Advanced**: Change objectives, add constraints
4. **Expert**: Extend to multi-scenario, MPC, etc.

---

## 💡 Key Concepts

- **Optimal Control**: Find best control inputs over time
- **Backward Euler**: Simple method to discretize ODEs
- **PI Control**: Proportional-Integral feedback controller
- **Nonlinear Programming**: Optimization with nonlinear constraints

---

## 📖 Further Reading

- `README_PYOMO.md` - Complete usage guide
- `PYOMO_CONVERSION_SUMMARY.md` - Detailed explanation
- `cstr_tutorial.ipynb` - Interactive learning

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Packages installed (`pyomo`, `numpy`, `pandas`, `matplotlib`)
- [ ] IPOPT solver installed
- [ ] Run the code successfully
- [ ] View the plots
- [ ] Understand the results
- [ ] Try modifying parameters

---

## 🆘 Need Help?

1. Check `README_PYOMO.md` for detailed documentation
2. Read error messages carefully
3. Verify all dependencies are installed
4. Try the Jupyter notebook version for step-by-step execution
5. Check parameter values are reasonable

---

**Happy Optimizing! 🎉**
