import time
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Single_Source.Coverage_Guided_Row_Selection import compute_overall_coverage, compute_overall_penalty, coverage_guided_row_selection, optimize_selection, penalty_optimization
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources


def multi_source_algorithm_stat(sources, UR,theta,method="algo_main"):
    """
    Multi-Source Table Construction using statistics for source selection.
    Selects sources based on which brings the most new (col, value) pairs from UR.

    Returns: T (output DataFrame), num_sources_used, chosen_order (indices)
    """

    # Step 0: Prepare source statistics (summaries only)
    value_index, source_vectors = compute_UR_value_frequencies_in_sources(sources, UR)
    already_covered = set()
    chosen_order = []
    T = pd.DataFrame()
    i = 0
    final_cov = 0

    def get_next_M(sources, value_index, source_vectors, already_covered):
        all_ur_pairs = set(value_index.keys())
        remaining_not_covered = all_ur_pairs - already_covered
        print(f"DEBUG: Searching for a new candidate. Remaining not covered: {remaining_not_covered}")
        max_new_covered = -1
        best_src_idx = None
        for idx, vector in source_vectors.items():
            new_covered = set()
            for (col, val), v_idx in value_index.items():
                if (col, val) in already_covered:
                    continue
                if vector[v_idx] > 0:
                    new_covered.add((col, val))
            if len(new_covered) > max_new_covered:
                max_new_covered = len(new_covered)
                best_src_idx = idx
        if best_src_idx is None or max_new_covered == 0:
            return None
        return best_src_idx   # <----- Return index, not DataFrame!

   
    while True:
        src_idx = get_next_M(sources, value_index, source_vectors, already_covered)
        if src_idx is None:
            break
        M_i = sources[src_idx]   # This is now the DataFrame
        chosen_order.append(src_idx + 1)  # 1-based for reporting
        common_cols = [col for col in UR.columns if col in M_i.columns and col != "Identifiant"]
        if not common_cols:
            i += 1
            continue

    
        new_T, _ = coverage_guided_row_selection(M_i, UR, theta)
        # Union tables
        if T.empty:
            T = new_T
        elif not new_T.empty:
            T = T.set_index("Identifiant").combine_first(new_T.set_index("Identifiant")).reset_index()
        # Update covered pairs using stats, NOT by peeking into M_i
        vector = source_vectors[src_idx]
        for (col, val), v_idx in value_index.items():
            if vector[v_idx] > 0:
                already_covered.add((col, val))
        # Check current coverage
        final_cov, _ = compute_overall_coverage(T, UR)
        if final_cov >= theta:
            break
        i += 1
        print(f"Adding source (index {src_idx+1}) with {len(M_i)} rows")
        print(f"Current chosen_order: {chosen_order}")
        print(f"Table shape after merge: {T.shape}")
        

    # Optional: Penalty optimization
    if method == "coverage_penalty" and final_cov >= theta:
        T, _ = penalty_optimization(T, pd.concat(sources, ignore_index=True), UR, i, theta)
    if method == "algo_main" and final_cov >= theta:
        T, _ = penalty_optimization(T, pd.concat(sources, ignore_index=True), UR, i, theta)
        T, _ = optimize_selection(T, UR)
    
    return T, len(chosen_order), chosen_order


def main():
    
    from helpers.test_cases import TestCases
    from helpers.Source_Constructors import SourceConstructor

    # Load test data
    T, UR = TestCases().get_case(23)
    constructor = SourceConstructor(T, UR, seed=42)

    # Generate sources (any strategy works)
    sources = constructor.low_penalty_sources()

# Save to data/source_0.csv, source_1.csv, ...
    for i, df in enumerate(sources):
        df.to_csv(f"data/source_{i}.csv", index=False)

    # # Load test case data
    # from helpers.test_cases import TestCases
    # test_cases = TestCases()
    # T_input, UR = test_cases.get_case(21)     # Or another case number

    # # Choose theta (coverage requirement)
    # theta = 1.0

    # # Choose source construction strategy
    # from helpers.Source_Constructors import SourceConstructor
    # constructor = SourceConstructor(T_input, UR, seed=42) 
    # sources = constructor.high_penalty_sources()
    
    # # Now run the multi-source algorithm
    # from Single_Source.Coverage_Guided_Row_Selection import (
    #     algo_main, compute_overall_coverage, optimize_selection, penalty_optimization
    # )
    # from helpers.statistics_computation import compute_UR_value_frequencies_in_sources
    # start_time = time.time()
    # T_output, num_used, chosen_order = multi_source_algorithm_stat(sources,UR,theta, method="algo_main")
    # elapsed = time.time() - start_time
    # # Show results
    # final_cov, _ = compute_overall_coverage(T_output, UR)
    # final_pen, _ = compute_overall_penalty(T_output, UR)
    # print(f"\nFinal Coverage: {final_cov:.4f}")
    # print(f"Final Penalty: {final_pen:.4f}\n")
    # print(f"Elapsed time: {elapsed:.2f} seconds")
    # # Optional: print penalty if desired
    # # final_pen, _ = compute_overall_penalty(T_output, UR)
    # # print(f"Final Penalty: {final_pen:.4f}")
    # print(f"Final Table:\n{T_output}\n")
    # print(f"Number of sources used: {num_used}")
    # print(f"Order of sources used (1-based): {chosen_order}")

# To run as a script:
if __name__ == "__main__":
    main()