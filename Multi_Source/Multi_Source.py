import pandas as pd

from Single_Source.Coverage_Guided_Row_Selection import algo_main, compute_overall_coverage, compute_overall_penalty, coverage_guided_row_selection, optimize_selection, penalty_optimization
from helpers.Source_Constructors import SourceConstructor
from helpers.Source_Constructors import dataframe_to_ur_dict
from helpers.T_splitter_into_M import split_by_columns, split_by_diagonal, split_by_hybrid, split_by_keywords, split_by_overlapping_rows, split_by_rows
from helpers.test_cases import TestCases

def get_next_M(sources, i):
    """ Get the next source M_i from the set of sources M.
    """
    if i < len(sources):
        return sources[i]
    return None  

def multi_source_algorithm(sources, UR, theta, method="algo_main"):
    T = pd.DataFrame()
    chosen_order = []
    i = 0
    terminate = False
    
    # STEP 1: Incrementally select sources with coverage_guided_row_selection
    while not terminate:
        terminate = True
        M_i = get_next_M(sources, i)
        if M_i is None:
            break  # No more sources
        common_cols = [col for col in UR.columns if col in M_i.columns and col != "Identifiant"]
        chosen_order.append(i + 1)
        if not common_cols:
            i += 1
            terminate = False
            continue
        new_T, _ = coverage_guided_row_selection(M_i, UR, theta)
        if T.empty:
            T = new_T
        elif not new_T.empty:
            T = T.set_index("Identifiant").combine_first(new_T.set_index("Identifiant")).reset_index()
        final_cov, _ = compute_overall_coverage(T, UR)
        if final_cov >= theta:
            break
        i += 1
        terminate = False

    # STEP 2: If coverage_penalty, run penalty_optimization at the end
    if method == "coverage_penalty" and final_cov >= theta:
        T, _ = penalty_optimization(T, pd.concat(sources, ignore_index=True), UR, 0, theta)

    # STEP 3: If algo_main, run optimize_selection at the end
    if method == "algo_main" and final_cov >= theta:
        T, _ = penalty_optimization(T, pd.concat(sources, ignore_index=True), UR, 0, theta)
        T, _ = optimize_selection(T, UR)
        print(T)

    return T, i + 1, chosen_order




# --- Main Function ---
def main():
    """ Main function to split T and run the multi-source algorithm. """
    # Load the test case
    test_cases = TestCases()
    T_input, UR = test_cases.get_case(23)  # Load predefined test case 1
    #UR = dataframe_to_ur_dict(UR)
    theta = 1  # Example coverage threshold
    constructor = SourceConstructor(T_input, UR)
    #sources = constructor.low_coverage_sources()
    #sources = constructor.group_by_sources()
    #sources = constructor.low_penalty_sources()
    sources = constructor.high_penalty_sources()
    
    #print(f"✅ Created {len(sources)} high-penalty sources.")

    # print(f"Generated {len(sources)} sources")
    # for i, src in enumerate(sources[:5]):
    #     print(f"Source {i} (first few rows):")
    #     print(src.head())

    # --- Choose one split method ---
    #Same schema:
    #sources = split_by_diagonal(T_input) 
    #sources = split_by_rows(T_input) 
    #sources = split_by_overlapping_rows(T_input, overlap_size=5) 

    #Different schema:
    #sources = split_by_columns(T_input)  
    #sources = split_by_hybrid(T_input)  
    #sources = split_by_keywords(T_input)
    
    
    T_output, _, _ = multi_source_algorithm(sources, UR, theta, method="sss") 
    T_output, _ = optimize_selection(T_output, UR)

    # Compute final coverage
    final_cov, _ = compute_overall_coverage(T_output, UR)
    final_pen, _ = compute_overall_penalty(T_output, UR)

    # Display results
    print(f"Final Coverage: {final_cov:.4f}")
    print(f"Final Penalty: {final_pen:.4f}")
    print(f"Final Table:\n{T_output}\n")

if __name__ == "__main__":
    main()