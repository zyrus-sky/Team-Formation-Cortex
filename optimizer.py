import pandas as pd
from ortools.sat.python import cp_model

def find_optimal_team(employees_df, project_fit_scores, required_skills, max_budget_per_hour):
    """
    Finds the optimal project team based on multiple objectives using Google OR-Tools.

    Args:
        employees_df: DataFrame containing all employee data, including performance.
        project_fit_scores: A dictionary mapping EmployeeID to their relevance score for the new project.
        required_skills: A list of mandatory skills for the project.
        max_budget_per_hour: The maximum hourly cost for the entire team.

    Returns:
        A tuple containing the list of recommended employee IDs and a dictionary with analysis.
    """
    model = cp_model.CpModel()

    # --- 1. Create Decision Variables ---
    # For each available employee, create a boolean variable: 1 if chosen, 0 if not.
    employee_vars = {
        row['EmployeeID']: model.NewBoolVar(f"emp_{row['EmployeeID']}")
        for index, row in employees_df[employees_df['Availability'] == 'Available'].iterrows()
    }

    # --- 2. Define Constraints ---
    # Constraint 1: All required skills must be covered by the team.
    for skill in required_skills:
        employees_with_skill = [
            employee_vars[emp_id]
            for emp_id, var in employee_vars.items()
            if skill in employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Skills'].iloc[0]
        ]
        if employees_with_skill:
            model.Add(sum(employees_with_skill) >= 1)

    # Constraint 2: The total team cost must not exceed the budget.
    total_cost = sum(
        employee_vars[emp_id] * employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Cost_per_Hour'].iloc[0]
        for emp_id in employee_vars
    )
    model.Add(total_cost <= max_budget_per_hour)
    
    # Constraint 3: Ensure average team age is above 30 (for experience)
    team_size = sum(employee_vars.values())
    total_age = sum(
        employee_vars[emp_id] * employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Age'].iloc[0]
        for emp_id in employee_vars
    )

    # To handle the conditional constraint correctly
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
    
    # Weights to prioritize different factors
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
             return None, {"status": "No feasible team found. The constraints (budget, skills, age) might be too restrictive."}

        final_cost = sum(employees_df.loc[employees_df['EmployeeID'] == emp_id, 'Cost_per_Hour'].iloc[0] for emp_id in recommended_ids)
        final_fit_score = sum(project_fit_scores.get(emp_id, 0) for emp_id in recommended_ids)
        
        analysis = { "status": "Optimal team found", "total_cost": final_cost, "avg_fit_score": final_fit_score / len(recommended_ids) if recommended_ids else 0 }
        return recommended_ids, analysis
    else:
        return None, {"status": "No feasible team found for the given constraints."}

