from ortools.sat.python import cp_model
import pandas as pd

def find_optimal_team(
    employees_df: pd.DataFrame,
    project_fit_scores: dict,
    required_skills: list,
    max_budget_per_hour: float
):
    """
    Finds the optimal project team based on multiple objectives using Google OR-Tools.
    This version operates on a pre-filtered DataFrame of candidates for better performance.
    """
    model = cp_model.CpModel()

    # --- 1. Create Decision Variables ---
    # The incoming employees_df is already filtered for availability and at least one skill.
    employee_vars = {
        row['EmployeeID']: model.NewBoolVar(f"emp_{row['EmployeeID']}")
        for index, row in employees_df[employees_df['Availability'] == 'Available'].iterrows()
    }

    if not employee_vars:
        return None, {"status": "No available employees match the initial skill filter."}

    # --- 2. Define Constraints ---
    
    # Constraint 1: All required skills must be covered by the selected team.
    for skill in required_skills:
        employees_with_skill = [
            employee_vars[emp_id]
            for emp_id in employee_vars
            # Ensure the skill is present in the employee's skill list
            if skill in str(employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Skills'].iloc[0])
        ]
        if employees_with_skill:
            model.Add(sum(employees_with_skill) >= 1)

    # Constraint 2: The total team cost must not exceed the budget.
    total_cost = sum(
        employee_vars[emp_id] * employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Cost_per_Hour'].iloc[0]
        for emp_id in employee_vars
    )
    model.Add(total_cost <= max_budget_per_hour)
    
    # Constraint 3: Ensure average team age is above 30 (encourages experience)
    team_size = sum(employee_vars.values())
    total_age = sum(
        employee_vars[emp_id] * employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Age'].iloc[0]
        for emp_id in employee_vars
    )

    is_team_non_empty = model.NewBoolVar('is_team_non_empty')
    model.Add(team_size > 0).OnlyEnforceIf(is_team_non_empty)
    model.Add(team_size == 0).OnlyEnforceIf(is_team_non_empty.Not())
    model.Add(total_age >= 30 * team_size).OnlyEnforceIf(is_team_non_empty)
    
    # --- 3. Define the Multi-Objective Function ---
    SCALE_FACTOR = 100 
    
    total_fit_score = sum(
        employee_vars[emp_id] * int(project_fit_scores.get(emp_id, 0) * SCALE_FACTOR)
        for emp_id in employee_vars
    )
    total_performance_score = sum(
        employee_vars[emp_id] * int(employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Avg_Performance'].iloc[0] * SCALE_FACTOR)
        for emp_id in employee_vars
    )
    
    fit_weight = 3
    performance_weight = 2
    
    model.Maximize((fit_weight * total_fit_score) + (performance_weight * total_performance_score))

    # --- 4. Solve the Model ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # --- 5. Extract and Return Results ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        recommended_ids = [emp_id for emp_id, var in employee_vars.items() if solver.Value(var) == 1]
        
        if not recommended_ids:
             return None, {"status": "No feasible team could be formed with the given constraints (e.g., budget, skill combinations)."}

        final_cost = sum(employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Cost_per_Hour'].iloc[0] for emp_id in recommended_ids)
        final_fit_score = sum(project_fit_scores.get(emp_id, 0) for emp_id in recommended_ids)
        
        analysis = { "status": "Optimal team found", "total_cost": final_cost, "avg_fit_score": final_fit_score / len(recommended_ids) if recommended_ids else 0 }
        return recommended_ids, analysis
    else:
        return None, {"status": "Optimizer could not find a solution for the given constraints."}

