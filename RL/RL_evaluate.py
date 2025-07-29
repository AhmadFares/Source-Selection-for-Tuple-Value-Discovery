from contextlib import contextmanager
import os
import time
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from Multi_Source.Multi_Source import multi_source_algorithm
from Multi_Source.Multi_Source_Cov_Stats import multi_source_algorithm_stat

import random
np.random.seed(42)
random.seed(42)
from RL.RL_Env import DataSelectionEnv
from Single_Source.Coverage_Guided_Row_Selection import (
    compute_overall_coverage,
    compute_overall_penalty,
    optimize_selection,
)
from helpers.Source_Constructors import SourceConstructor
from helpers.T_splitter_into_M import split_uniform_by_rows
from helpers.test_cases import TestCases
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources

# @contextmanager
# def suppress_output():
#     with open(os.devnull, "w") as devnull:
#         old_stdout = sys.stdout
#         old_stderr = sys.stderr
#         sys.stdout = devnull
#         sys.stderr = devnull
#         try:
#             yield
#         finally:
#             sys.stdout = old_stdout
#             sys.stderr = old_stderr

def evaluate_offline(method, sources_list, UR, theta):
    start_time = time.time()
    T_out, _, chosen_order = multi_source_algorithm(sources_list, UR, theta, method=method)
    T_out, _ = optimize_selection(T_out, UR)
    elapsed = time.time() - start_time
    final_cov, _ = compute_overall_coverage(T_out, UR)
    final_pen, _ = compute_overall_penalty(T_out, UR)
    print(T_out)
    return {
        "coverage": final_cov,
        "penalty": final_pen,
        "time": elapsed,
        "order": chosen_order,
        "steps": len(chosen_order),
    }

def evaluate_rl_agent(model_path, sources_list, UR, alpha, beta, gamma):
    value_index, source_stats = compute_UR_value_frequencies_in_sources(sources_list, UR)
    #zero_stats = {i: np.zeros_like(v) for i, v in source_stats.items()}
    env = DataSelectionEnv(
        sources_list, UR, statistics=source_stats, value_index=value_index,
        alpha=alpha, beta=beta, gamma=gamma
    )
    model = DQN.load(model_path)
    start_time = time.time()
    obs = env.reset()
    done = False
    chosen_order = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        chosen_order.append(action)
        obs, reward, done, info = env.step(action)
    elapsed = time.time() - start_time
    env.current_table = optimize_selection(env.current_table, env.UR)[0]
    final_cov_rl, _ = compute_overall_coverage(env.current_table, env.UR)
    final_pen_rl, _ = compute_overall_penalty(env.current_table, env.UR)
    print(env.current_table)
    return {
        "coverage": final_cov_rl,
        "penalty": final_pen_rl,
        "time": elapsed,
        "order": list(env.selected_sources),
        "steps": len(env.selected_sources),
        
    }

# ===================== MAIN SCRIPT ===========================

test_cases = TestCases()
test_cases.load_mathe_case()
T, UR = test_cases.get_case(23)
constructor = SourceConstructor(T, UR, seed=42)  # Seed for reproducibility!
#sources_list= split_uniform_by_rows(T, 10)
#sources_list = constructor.high_penalty_sources()
#sources_list = constructor.low_coverage_sources()
sources_list = constructor.low_penalty_sources()

theta = 1.0
                                              
results = []

# ------- OFFLINE METHODS -------
offline_methods = [
     ("coverage_only", "Offline Coverage Only"),
     ("coverage_penalty", "Offline Coverage+Penalty"),
   ("algo_main", "Offline Full Algo Main"),
    ("stats", "Offline Stats-based Multi-Source"), 
]

for method, label in offline_methods:
    print(f"\n--- Evaluating {label} ---")
    if method == "stats":
        start_time = time.time()
        T_out, _, chosen_order = multi_source_algorithm_stat(sources_list, UR, theta, method="algo_main")
        T_out, _ = optimize_selection(T_out, UR)
        print(T_out)
        elapsed = time.time() - start_time
        final_cov, _ = compute_overall_coverage(T_out, UR)
        final_pen, _ = compute_overall_penalty(T_out, UR)
        order = chosen_order
        steps = len(chosen_order)
        time_spent = elapsed
    else:
        res = evaluate_offline(method, sources_list, UR, theta)
        final_cov = res["coverage"]
        final_pen = res["penalty"]
        order = res["order"]
        steps = res["steps"]
        time_spent = res["time"]
    results.append({
        "variant": label,
        "coverage": final_cov,
        "penalty": final_pen,
        "steps": steps,
        "time": time_spent,
        "order": order,
        "type": "offline",
    })

# ------- RL AGENTS -------
rl_agents = [
    #("dqn_model_low_coverage_2.zip", "RL Low Coverage"),
    ("low_penalty_23_dqn_model", "RL Low Penalty"),
     ("high_penalty_23_dqn_model", "RL High Penalty"),
    ("low_coverage_23_dqn_model", "RL Low Coverage")
]

alpha = 0.6
beta = 0.3
gamma = 0.1

for path, label in rl_agents:
    print(f"\n--- Evaluating RL agent: {label} ---")
    print('URRRR')
    print(UR)
    res = evaluate_rl_agent(path, sources_list, UR, alpha, beta, gamma)
    results.append({
        "variant": label,
        "coverage": res["coverage"],
        "penalty": res["penalty"],
        "steps": res["steps"],
        "time": res["time"],
        "order": res["order"],
        "type": "rl",
    })

# ------------ Save as DataFrame / CSV ---------------
df = pd.DataFrame(results)
# Convert list to string for CSV
df["order"] = df["order"].apply(lambda x: ",".join(map(str, x)))
df.to_csv("evaluation_summary.csv", index=False)
print("\n=== SUMMARY TABLE ===\n")
print(df)
