import numpy as np
import pandas as pd
import random
import gym
from gym.utils import seeding
from gym import spaces

from Single_Source.Coverage_Guided_Row_Selection import algo_main, compute_overall_coverage, compute_overall_penalty, optimize_selection

class DataSelectionEnv(gym.Env):
    def __init__(self, sources_list, user_request, statistics, value_index,alpha=0.5, beta=0.3, gamma=0.2): 
        self.sources_list = sources_list  
        self.max_steps = len(self.sources_list)
        self.UR = user_request
        self.gamma = gamma
        self.statistics = statistics
        self.alpha = alpha
        self.beta = beta
        self.value_index = value_index
        self.obs_len = 2 + len(sources_list) + len(sources_list) * len(value_index)

        self.num_sources = len(self.sources_list)
        self.stop_action = self.num_sources # index of the stop action
        self.action_space = spaces.Discrete(self.num_sources + 1)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_len,),
            dtype=np.float32
        )

        # Ensures that every time the environment is initialized, it starts from a clean, consistent state
        self.reset()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.selected_sources = set()
        self.current_table = pd.DataFrame(columns=self.UR.columns.tolist() + ['Identifiant'])
        self.current_coverage, _ = compute_overall_coverage(self.current_table, self.UR)
        self.current_penalty, _ = compute_overall_penalty(self.current_table, self.UR)
        self.last_coverage = 0.0
        self.last_penalty = 0.0
        self.last_steps = 0
        self.steps_taken = 0

        return self.get_state()

    def step(self, action):
        action = int(np.asarray(action).item())
        
        if action == self.stop_action:
            done = True
            if len(self.selected_sources) == 0:  # Agent chooses STOP as first action
                reward = -10  # High penalty for stopping without any selection
            else:
                reward = self.compute_reward(self.current_coverage, self.current_penalty)
                #  Give bonus for stopping with full coverage
                if self.current_coverage >= 1.0:
                    reward += 50
                    
            info = {
                "stop": True,
                "coverage": self.current_coverage,
                "penalty": self.current_penalty,
                "steps": self.steps_taken,
                "selected_sources": list(self.selected_sources)
            }
            return self.get_state(), reward, done, info

        
        if action in self.selected_sources: #  Agent chooses a source it already selected
            reward = -10 # punish invalid repeat
            done = True
            self.steps_taken += 1
            info = {
                "stop": False,
                "coverage": self.current_coverage,
                "penalty": self.current_penalty,
                "steps": self.steps_taken,
                "error": f"Source {action} already selected."
            }
            return self.get_state(), reward, done, info

        #  Valid source selection
        self.selected_sources.add(action)
        self.steps_taken += 1

        selected_source = self.sources_list[action]
        new_T = algo_main(selected_source, self.UR, 1)
        if self.current_table.empty:
            self.current_table = new_T
        elif not new_T.empty:
            self.current_table = (
                self.current_table.set_index("Identifiant")
                .combine_first(new_T.set_index("Identifiant"))
                .reset_index()
            )
            optimized_table, _ = optimize_selection(self.current_table, self.UR)
            self.current_table = optimized_table
           # print(f"Added {len(new_T)} rows from source {action}.")
           # print(f"Current table size: {len(self.current_table)} rows.")

        new_coverage, _ = compute_overall_coverage(self.current_table, self.UR)
        new_penalty, _ = compute_overall_penalty(self.current_table, self.UR)

        reward = self.compute_reward(new_coverage, new_penalty)
        #print("CURRENT REWARD", reward)

        # update state tracking
        self.current_coverage = new_coverage
        self.current_penalty = new_penalty
        self.last_coverage = new_coverage
        self.last_penalty = new_penalty
        self.last_steps = self.steps_taken

        done = False
        next_state = self.get_state()
        info = {
            "stop": False,
            "coverage": self.current_coverage,
            "penalty": self.current_penalty,
            "steps": self.steps_taken,
            "selected_sources": list(self.selected_sources)
        }
        return next_state, reward, done, info


    def compute_reward(self, new_coverage, new_penalty):
        normalized_steps = self.steps_taken / self.max_steps if self.max_steps > 0 else 0.0
        reward = (
            self.alpha * new_coverage -
            self.beta * new_penalty -
            self.gamma * normalized_steps
        )
        return 10 * reward
# def compute_delta_reward(self, new_coverage, new_penalty):
#     delta_coverage = new_coverage - self.last_coverage
#     delta_penalty = new_penalty - self.last_penalty
#     delta_steps = self.steps_taken - self.last_steps

#     reward = (
#         self.alpha * delta_coverage
#         - self.beta * delta_penalty
#         - self.gamma * delta_steps
#     )
#     return reward

    def get_state(self):
        state = []

        # Global metrics
        state += [self.current_coverage, self.current_penalty]

        # Selection mask
        selected_mask = [1 if i in self.selected_sources else 0 for i in range(len(self.sources_list))]
        state += selected_mask

        # Precomputed stats per source
        for i in range(len(self.sources_list)):
            state += list(self.statistics.get(i, np.zeros(len(self.statistics[0]), dtype=np.float32)))

        return np.array(state, dtype=np.float32)


    # def get_available_actions(self):
    #     return [i for i in range(len(self.sources_list)) if i not in self.selected_sources]