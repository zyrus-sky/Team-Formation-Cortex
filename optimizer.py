from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def find_optimal_team(
    employees_df: pd.DataFrame,
    project_fit_scores: Dict[int, float],
    required_skills: List[str],
    max_budget_per_hour: float,
    min_avg_age: float = 30.0,
    fit_weight: int = 3,
    performance_weight: int = 2,
    max_team_size: Optional[int] = None,
    time_limit_s: int = 10,
    workers: int = 8,
    soft_skills: Optional[List[str]] = None,   # NEW: optional soft skills
    soft_penalty: int = 1,                     # NEW: penalty weight per missing soft skill
) -> Tuple[List[int], Dict]:
    """
    CP-SAT model with soft skills:
    - Hard constraints: budget, min avg age, optional team size, required (canonical) skills.
    - Soft skills: preferred tags; each missing tag incurs a penalty in the objective (no infeasible fail).
    Objective: maximize fit_weight*Fit + performance_weight*Avg_Performance - soft_penalty*misses.
    """
    # ---------- Defensive copies and coercions ----------
    df = (employees_df or pd.DataFrame()).copy()
    if df.empty or "Availability" not in df.columns:
        return [], {"status": "No employees or missing 'Availability' column."}

    available = df[df["Availability"] == "Available"].reset_index(drop=True)
    if available.empty:
        return [], {"status": "No available employees."}

    # Ensure key numeric columns exist and are numeric
    for col in ["Cost_per_Hour", "Age"]:
        if col not in available.columns:
            available[col] = 0.0
        available[col] = pd.to_numeric(available[col], errors="coerce").fillna(0.0)

    if "Avg_Performance" not in available.columns:
        available["Avg_Performance"] = 0.0
    available["Avg_Performance"] = pd.to_numeric(available["Avg_Performance"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if "Skills" not in available.columns:
        available["Skills"] = ""

    # Vectors
    emp_ids = available["EmployeeID"].astype(int).tolist()
    costs = available["Cost_per_Hour"].astype(float).to_numpy()
    ages = available["Age"].astype(float).to_numpy()
    skills_text = available["Skills"].astype(str).str.lower().tolist()
    perf = available["Avg_Performance"].astype(float).clip(0.0, 1.0).to_numpy()

    # Fit scores (map missing to 0)
    pfit = available["EmployeeID"].map(project_fit_scores if isinstance(project_fit_scores, dict) else {}).fillna(0.0)
    fit = pfit.astype(float).to_numpy()

    # ---------- Model ----------
    model = cp_model.CpModel()
    n = len(available)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    # Budget in cents
    budget_cents = int(round(float(max(0.0, max_budget_per_hour)) * 100))
    cost_cents = [int(round(float(costs[i]) * 100)) for i in range(n)]
    model.Add(sum(cost_cents[i] * x[i] for i in range(n)) <= budget_cents)

    # Average age >= threshold => sum((age - tau)*x) >= 0 scaled
    age_shift = [int(round((float(ages[i]) - float(min_avg_age)) * 100)) for i in range(n)]
    model.Add(sum(age_shift[i] * x[i] for i in range(n)) >= 0)

    # Team size cap
    if max_team_size is not None and int(max_team_size) > 0:
        model.Add(sum(x) <= int(max_team_size))

    # Required skills (hard). Normalize to lowercase and strip.
    req = [str(s).strip().lower() for s in (required_skills or []) if str(s).strip()]
    for s in req:
        cover = [1 if s in skills_text[i] else 0 for i in range(n)]
        if sum(cover) == 0:
            # Keep original behavior: if a truly required skill isn't present at all, fail fast.
            return [], {"status": f"Required skill '{s}' not found in workforce"}
        model.Add(sum(cover[i] * x[i] for i in range(n)) >= 1)

    # ---------- Objective ----------
    SCALE = 1000
    base_obj_terms = []
    for i in range(n):
        fit_part = fit_weight * int(round(float(fit[i]) * SCALE))
        perf_part = performance_weight * int(round(float(perf[i]) * SCALE))
        base_obj_terms.append((fit_part + perf_part) * x[i])

    # Soft-skill penalties (optional)
    penalty_terms = []
    soft = [str(s).strip().lower() for s in (soft_skills or []) if str(s).strip()]
    if soft:
        # For each soft skill s: if no selected member has s, incur a penalty
        for idx, s in enumerate(soft):
            cover = [1 if s in skills_text[i] else 0 for i in range(n)]
            if sum(cover) == 0:
                # If no one in workforce has s at all, there's nothing to penalize; skip
                continue
            covered = model.NewBoolVar(f"soft_cov_{idx}")
            # covered == 1 if sum(cover[i]*x[i]) >= 1
            # Use reified constraints with big-M trick via two inequalities
            # Implement via implication:
            # If covered == 1 -> sum >= 1
            # If covered == 0 -> sum <= 0
            # Using OnlyEnforceIf is supported in OR-Tools CP-SAT
            model.Add(sum(cover[i] * x[i] for i in range(n)) >= 1).OnlyEnforceIf(covered)
            model.Add(sum(cover[i] * x[i] for i in range(n)) <= 0).OnlyEnforceIf(covered.Not())
            penalty_terms.append(soft_penalty * SCALE * (1 - covered))

    # Maximize base objective minus penalties
    model.Maximize(sum(base_obj_terms) - (sum(penalty_terms) if penalty_terms else 0))

    # ---------- Solve ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], {"status": f"Solver status {status}"}

    chosen_idx = [i for i in range(n) if solver.Value(x[i]) == 1]
    chosen_ids = [int(emp_ids[i]) for i in chosen_idx]
    if not chosen_ids:
        return [], {"status": "No feasible team under constraints"}

    analysis = {
        "status": "ok",
        "objective": float(solver.ObjectiveValue()) / SCALE,
        "total_cost_per_hour": float(np.sum([costs[i] for i in chosen_idx])) if chosen_idx else 0.0,
        "avg_age": float(np.mean([ages[i] for i in chosen_idx])) if chosen_idx else 0.0,
        "required_skills": req,
        "soft_skills": soft,
        "fit_weight": int(fit_weight),
        "performance_weight": int(performance_weight),
        "soft_penalty": int(soft_penalty),
        "solver_time_s": float(solver.WallTime()),
        "solver_status": int(status),
        "team_size": int(len(chosen_ids)),
    }
    return chosen_ids, analysis
