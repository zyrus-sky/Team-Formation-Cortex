from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def find_optimal_team(
    employees_df: pd.DataFrame,
    project_fit_scores: Dict[int, float],
    required_skills: List[str],
    max_budget_per_hour: float,
    min_avg_age: float = 30.0,
    fit_weight: int = 3,
    performance_weight: int = 2,
    max_team_size: int | None = None,
    time_limit_s: int = 10,
    workers: int = 8,
) -> Tuple[List[int], Dict]:
    """
    CP-SAT model:
    - Cover all required skills at least once
    - Sum(cost) <= max_budget_per_hour
    - Average age >= min_avg_age
    - Optional: team size cap
    Objective: maximize fit_weight*Fit + performance_weight*Avg_Performance
    """
    df = employees_df.copy()
    available = df[df["Availability"] == "Available"].reset_index(drop=True)
    if available.empty:
        return [], {"status": "No available employees."}

    emp_ids = available["EmployeeID"].astype(int).tolist()
    costs = available["Cost_per_Hour"].astype(float).to_numpy()
    ages = available["Age"].astype(float).to_numpy()
    skills_text = available["Skills"].astype(str).str.lower().tolist()
    perf = available.get("Avg_Performance", pd.Series([0.0]*len(available))).astype(float).clip(0.0, 1.0).to_numpy()
    fit = available["EmployeeID"].map(project_fit_scores).fillna(0.0).astype(float).to_numpy()

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(available))]

    # Budget (use cents to keep integers)
    model.Add(sum(int(round(costs[i]*100)) * x[i] for i in range(len(x))) <= int(round(max_budget_per_hour*100)))

    # Average age >= threshold => sum((age - tau)*x) >= 0
    age_shift = [int(round((ages[i] - min_avg_age) * 100)) for i in range(len(x))]
    model.Add(sum(age_shift[i] * x[i] for i in range(len(x))) >= 0)

    # Team size cap (optional)
    if max_team_size is not None and max_team_size > 0:
        model.Add(sum(x) <= int(max_team_size))

    # Skill coverage
    req = [s.strip().lower() for s in required_skills if str(s).strip()]
    for s in req:
        mask = [1 if s in skills_text[i] else 0 for i in range(len(x))]
        if sum(mask) == 0:
            return [], {"status": f"Required skill '{s}' not found in workforce"}
        model.Add(sum(mask[i] * x[i] for i in range(len(x))) >= 1)

    # Objective
    SCALE = 1000
    obj = sum((fit_weight*int(round(fit[i]*SCALE)) + performance_weight*int(round(perf[i]*SCALE))) * x[i] for i in range(len(x)))
    model.Maximize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = workers

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], {"status": f"Solver status {status}"}

    chosen_idx = [i for i in range(len(x)) if solver.Value(x[i]) == 1]
    chosen_ids = [int(emp_ids[i]) for i in chosen_idx]
    if not chosen_ids:
        return [], {"status": "No feasible team under constraints"}

    analysis = {
        "status": "ok",
        "objective": float(solver.ObjectiveValue())/SCALE,
        "total_cost_per_hour": float(np.sum(costs[chosen_idx])),
        "avg_age": float(np.mean(ages[chosen_idx])),
        "required_skills": req,
        "fit_weight": fit_weight,
        "performance_weight": performance_weight,
        "solver_time_s": float(solver.WallTime()),
        "solver_status": int(status),
        "team_size": int(len(chosen_ids)),
    }
    return chosen_ids, analysis
