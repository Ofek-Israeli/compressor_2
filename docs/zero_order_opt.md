# Low-$d$, Expensive-Evaluation Zero-Order Optimization (Summary + Python Usage)

Setting: minimize a black-box objective $f:\mathbb{R}^d\to\mathbb{R}$ where each evaluation of $f(x)$ is expensive and $d$ is small (e.g., $\lesssim 20$--$50$).

## Methods to Know (beyond GP Bayesian Optimization)

| Method | Typical inputs | Typical outputs | Python library |
|---|---|---|---|
| **(GP-)Bayesian Opt.** | $f(x)$; box bounds; eval budget; acquisition rule; noise model (optional) | Best found point $x^{*}$, value $f(x^{*})$, trial history, surrogate model (optional) | `scikit-optimize (skopt)` |
| **Trust-region DFO** (local surrogate) | $f(x)$; initial point $x_0$; bounds/constraints (optional); trust-region / max evals | Local optimum $x^{*}$, $f(x^{*})$, diagnostics (iters/evals) | `pdfo` (Powell solvers: NEWUOA/BOBYQA/etc.) |
| **RBF surrogate global opt.** | $f(x)$; bounds; eval budget; variable types (cont./int/cat) | Best found point $x^{*}$, $f(x^{*})$, diagnostics | `rbfopt` |
| **DIRECT** (deterministic global partitioning) | $f(x)$; bounds; eval budget / tolerances | Best found point $x^{*}$, $f(x^{*})$, eval count | `scipy.optimize.direct` |
| **TPE** (BO alternative) | Objective defined via “suggest” calls; search space; number of trials | Best trial parameters, best objective value, full trial log | `optuna` |
| **SMAC** (RF surrogate BO alternative) | Objective over a ConfigSpace; scenario (trial budget, determinism, etc.) | Incumbent configuration (best found), run history | `smac` (SMAC3) |
| **Hybrid: global $\rightarrow$ local refine** | Global method + local method; bounds; budgets split across stages | Refined $x^{*}$ with fewer wasted evals | Combine: `scipy.direct/skopt/optuna/smac` + `pdfo` |

---

## Python Usage Cheatsheet (minimal)

### 1) (GP-)Bayesian Optimization via `skopt`

```python
# pip install scikit-optimize
from skopt import gp_minimize

def f(x):
    # x is a list-like of length d
    return (x[0] - 0.3)**2 + 0.1

res = gp_minimize(
    func=f,
    dimensions=[(-2.0, 2.0)],   # bounds per dimension
    acq_func="EI",
    n_calls=30,
    n_random_starts=5,
    random_state=0,
)

x_star, f_star = res.x, res.fun
print("x* =", x_star, "f(x*) =", f_star)
````

### 2) Trust-region DFO via `pdfo` (NEWUOA / BOBYQA)

```python
# pip install pdfo
import numpy as np
from pdfo import pdfo

# Unconstrained -> typically NEWUOA (or set method="newuoa")
def f(x):
    return float((x[0] - 0.3)**2 + (x[1] + 0.2)**2)

x0 = np.array([0.0, 0.0])
res = pdfo(f, x0, method="newuoa", options={"maxfev": 200})
print(res.x, res.fun, res.nfev)

# Bound-constrained -> typically BOBYQA (or set method="bobyqa")
from scipy.optimize import Bounds
bounds = Bounds([-1.0, -1.0], [1.0, 1.0])
res = pdfo(f, x0, method="bobyqa", bounds=bounds, options={"maxfev": 200})
print(res.x, res.fun, res.nfev)
```

### 3) RBF surrogate global optimization via `rbfopt`

```python
# pip install rbfopt
import numpy as np
from rbfopt import RbfoptUserBlackBox, RbfoptSettings, RbfoptAlgorithm

def obj_funct(x):
    x = np.asarray(x)
    return float((x[0] - 0.3)**2 + (x[1] + 0.2)**2)

d = 2
lb = np.array([-1.0, -1.0], dtype=float)
ub = np.array([ 1.0,  1.0], dtype=float)
var_type = np.array(["R"] * d)   # "R"=real, "I"=integer, "C"=categorical

bb = RbfoptUserBlackBox(dimension=d, var_lower=lb, var_upper=ub,
                        var_type=var_type, obj_funct=obj_funct)

settings = RbfoptSettings(max_evaluations=200)  # evaluation budget
alg = RbfoptAlgorithm(settings, bb)

best_val, best_x, iters, evals, noisy_evals = alg.optimize()
print(best_x, best_val, evals)
```

### 4) DIRECT via `scipy.optimize.direct`

```python
# pip install scipy
from scipy.optimize import direct, Bounds

def f(x):
    x0, x1 = x
    return 0.5*(x0**4 - 16*x0**2 + 5*x0 + x1**4 - 16*x1**2 + 5*x1)

bounds = Bounds([-4.0, -4.0], [4.0, 4.0])

res = direct(f, bounds, maxiter=200, maxfun=2000)
print(res.x, res.fun, res.nfev)
```

### 5) TPE via `optuna`

```python
# pip install optuna
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    x = trial.suggest_float("x", -10.0, 10.0)
    return x**2

sampler = TPESampler()
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=100)

print(study.best_params, study.best_value)
```

### 6) SMAC (random-forest surrogate) via `smac`

```python
# pip install smac ConfigSpace
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

def train(config: Configuration, seed: int = 0) -> float:
    x = config["x"]
    return (x - 0.3)**2  # minimize

configspace = ConfigurationSpace({"x": (-2.0, 2.0)})

scenario = Scenario(configspace, deterministic=True, n_trials=100)
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()

print("best config:", incumbent)
```

### 7) Hybrid: global search $\rightarrow$ local trust-region refinement

```python
# Example: DIRECT to find a good basin, then pdfo to polish.
import numpy as np
from scipy.optimize import direct, Bounds
from pdfo import pdfo

def f(x):
    return float((x[0] - 0.3)**2 + (x[1] + 0.2)**2)

bounds = Bounds([-1.0, -1.0], [1.0, 1.0])
res_global = direct(f, bounds, maxfun=300)  # small global budget
x0 = np.array(res_global.x)

res_local = pdfo(f, x0, method="bobyqa", bounds=bounds, options={"maxfev": 200})
print("global:", res_global.x, res_global.fun)
print("refined:", res_local.x, res_local.fun)
```

---

## Rule of Thumb

* **Need global exploration with very few evaluations:** GP-BO (`skopt`) / TPE (`optuna`) / SMAC (`smac`) / DIRECT (`scipy`).
* **Have a decent starting point and want fast local gains:** trust-region DFO (e.g., NEWUOA/BOBYQA via `pdfo`).
* **Often best in practice:** do a small global stage, then local refinement (Hybrid).