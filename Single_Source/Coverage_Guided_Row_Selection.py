import sys
import pandas as pd
import time
import numpy as np
from helpers.test_cases import TestCases
import numpy as np
import random
np.random.seed(42)
random.seed(42)
def compute_attr_coverage(T, UR, col):
    """
    Compute the coverage for a single attribute (column).
    """
    if col not in T.columns:
        return 0  # If the column does not exist in T, return 0 coverage.

    t_values = set(T[col].dropna())
    ur_values = set(UR[col].dropna()) if col in UR.columns else set()

    if not ur_values:  # If UR has no values in this column, coverage is 1.
        return 1

    return len(t_values.intersection(ur_values)) / len(ur_values)


def compute_overall_coverage(T, UR): 
    """Compute the overall coverage as the average of the per-attribute coverages."""
    coverages = []
    for col in UR.columns:
         if col == "Identifiant":
          continue
         cov = compute_attr_coverage(T, UR, col)
         coverages.append(cov)
    overall_cov = sum(coverages) / len(coverages) if coverages else 1
    return overall_cov, coverages

def compute_attr_penalty(T, UR, col):
    t_values = set(T[col].dropna())
    if not t_values:
        return 0  # If T has no non-null values, penalty is 0.
    ur_values = set(UR[col].dropna()) if col in UR.columns else set()
    return len(t_values - ur_values) / len(t_values)

def compute_overall_penalty(T, UR):
    """Compute the overall penalty as the average penalty across all columns in T."""
    penalties = []
    for col in T.columns:
        if col == "Identifiant":
          continue
        p = compute_attr_penalty(T, UR, col)
        penalties.append(p)
    overall_penalty = sum(penalties) / len(penalties) if penalties else 0
    return overall_penalty, penalties

# --- Step 1: Coverage-Guided Row Selection ---
def coverage_guided_row_selection(input_table, UR, theta):
    """
    Selects rows to maximize coverage while staying below threshold theta.
    Outputs `T` and the index `i` where it stopped.
    """
    common_cols = [col for col in UR.columns if col in input_table.columns]
    
    if not common_cols:
        print("No common columns between UR and T. Returning empty DataFrame.")
        return pd.DataFrame(columns=input_table.columns), 0  # Return empty DataFrame
    
    if "Identifiant" in input_table.columns:
         common_cols = ["Identifiant"] + common_cols
    
    input_table = input_table[common_cols]  

    selected_rows = []
    curr_coverage = 0
    count = 0
    count_if = 0

    UR_values = {col: set(UR[col].dropna().values) for col in common_cols if col != "Identifiant"}

    for i, row in enumerate(input_table.itertuples(index=False, name=None)):
        row_dict = dict(zip(input_table.columns, row))

        # Simulate adding the row to selected_rows
        T_curr_values = {col: set(row_dict[col] for row_dict in selected_rows) for col in UR_values}
        T_curr_values = {col: vals | {row_dict[col]} for col, vals in T_curr_values.items()}  

        # Compute overall coverage using precomputed UR values
        coverages = [len(T_curr_values[col] & UR_values[col]) / len(UR_values[col]) if UR_values[col] else 1 for col in T_curr_values]
        cov = sum(coverages) / len(coverages) if coverages else 1

        if cov <= theta and cov > curr_coverage:
            count_if += 1
            selected_rows.append(row_dict)
            curr_coverage = cov
        else:
            if curr_coverage >= theta:
                return pd.DataFrame(selected_rows), i  #  Stop as soon as `theta` is reached

    return pd.DataFrame(selected_rows), len(input_table)  # Return full index if not stopped early

# --- Step 2: Optimize Penalty  ---
# --- Step 2: Optimize Penalty  ---
def penalty_optimization(T, input_table, UR, i, theta):
    """
    Attempts to replace rows in T using rows from input_table[i:], minimizing penalty while maintaining coverage.
    Only considers common columns between T and input_table.
    """
    count = 0
    curr_penalty, _ = compute_overall_penalty(T, UR)
    if curr_penalty == 0:
        return T, count  # Stop early if penalty is already zero

    # Determine common columns (excluding 'Identifiant') 
    common_cols = [col for col in T.columns if col in input_table.columns and col != "Identifiant"]

    # Convert input_table to list of dicts (fast lookup)
    input_list = input_table.to_dict(orient="records")
    
    for idx in range(i, len(input_list)): 
        count += 1
        new_row_dict = input_list[idx]  # Get new row as a dictionary

        for j in range(len(T)):  
            T_sub = T.copy()  # Keep full structure

            # 🔹 Replace only the row at index j, ensuring Identifiant is also updated
            for col in common_cols:
                T_sub.at[j, col] = new_row_dict[col]

            # 🔹 Restore correct 'Identifiant' from input_table
            if "Identifiant" in input_table.columns:
                T_sub.at[j, "Identifiant"] = new_row_dict["Identifiant"]
            
            # Compute new coverage & penalty
            sub_cov, _ = compute_overall_coverage(T_sub, UR)
            sub_penalty, _ = compute_overall_penalty(T_sub, UR)
            
            if (
            str(T.iloc[j]["Identifiant"]) == "18233" and 
            str(new_row_dict["Identifiant"]) in ["default_6923", "default_8687"]
            ):
                print(f"Trying to replace 18233 with {new_row_dict['Identifiant']}: coverage={sub_cov}, penalty={sub_penalty}, curr_penalty={curr_penalty}")
            if sub_cov >= theta and sub_penalty < curr_penalty:
                T = T_sub.copy()  # Update T only if a better replacement is found
                curr_penalty = sub_penalty
                

                if curr_penalty == 0:
                    return T, count  # Stop early if penalty reaches zero

    return T, count


#--- Step 3: Optimize Selection ---
def optimize_selection(T, UR):
    """Removes unnecessary rows while maintaining coverage."""
    
    orig_cov, _ = compute_overall_coverage(T, UR)
    changed = True
    count=0
    while changed:
        changed = False
        for idx in T.index.tolist():
            count=count+1
            T_sub = T.drop(index=idx).reset_index(drop=True)
            sub_cov, _ = compute_overall_coverage(T_sub, UR)
            if sub_cov == orig_cov:
                T = T_sub.copy()
                changed = True
                break  # Restart the loop since T has changed
    return T, count

# --- Main Algorithm ---
def algo_main(input_table, UR, theta):
    """
    Executes Coverage-Guided Row Selection with Penalty Optimization.
    """
    T, i = coverage_guided_row_selection(input_table, UR, theta)
    T, count = penalty_optimization(T, input_table, UR, i, theta)
    T, optcount = optimize_selection(T, UR)
    #print(f"[algo_main] Returned {len(T)} rows.")
    #print(f"[algo_main] Coverage: {compute_overall_coverage(T, UR)[0]}")
    #print(f"[algo_main] Penalty: {compute_overall_penalty(T, UR)[0]}")

    return T

# --- Execution ---
if __name__ == '__main__':
    case_number = 1
    if len(sys.argv) > 1:
        try:
            case_number = int(sys.argv[1])
        except ValueError:
            print("Invalid case number provided. Defaulting to case 1.")
    
    test_cases = TestCases()
    T_input, UR = test_cases.get_case(case_number)

    print(f"Running test case {case_number}:\nT_input =\n{T_input}\n\nUR =\n{UR}\n")
    
    theta = 1

    start_time = time.time()
    T_output = algo_main(T_input, UR, theta)
    time_with_optimization = time.time() - start_time

    final_cov, _ = compute_overall_coverage(T_output, UR)
    final_penalty, _ = compute_overall_penalty(T_output, UR)

    print("\nFinal output table T_output:")
    print(T_output)
    print(f"Overall Coverage: {final_cov}")
    print(f"Overall Penalty: {final_penalty}")
    print(f"Execution Time: {time_with_optimization:.4f} seconds")
    