import datetime
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import shutil



from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


from RL.RL_Env import DataSelectionEnv
from helpers.test_cases import TestCases
from helpers.Source_Constructors import SourceConstructor
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
    
    
class MetricLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricLoggerCallback, self).__init__(verbose)
        self.rewards = []
        self.coverages = []
        self.penalties = []
        self.actions = []
        self.steps = []
        self.stopped = []
        self.episode_actions = []
        self.q_values = []  


    def _on_step(self) -> bool:
        action = self.locals.get('actions')
        if action is not None:
            # Handles both single and vectorized envs
            if isinstance(action, (list, np.ndarray)) and len(action) == 1:
                action = action[0]
            self.episode_actions.append(action)
    
        obs = self.locals.get('new_obs')
        model = self.model
        if obs is not None and model is not None:
            obs_arr = np.array(obs)
            if len(obs_arr.shape) == 1:
                obs_arr = obs_arr[None, :]
            q_vals = model.q_net(torch.tensor(obs_arr, dtype=torch.float32).to(model.device))
            self.q_values.append(q_vals.detach().cpu().numpy()[0])

        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    self.rewards.append(self.locals['rewards'][idx])
                    info = self.locals['infos'][idx]
                    self.coverages.append(info.get("coverage", 0.0))
                    self.penalties.append(info.get("penalty", 0.0))
                    self.steps.append(info.get("steps", 0))
                    self.stopped.append(info.get("stop", False))
                    self.actions.append(self.episode_actions)
                    self.episode_actions = []

                    # LOG EVERY 500 EPISODES
                    if len(self.rewards) > 0 and len(self.rewards) % 500 == 0:
                        mean_reward = np.mean(self.rewards[-500:])
                        mean_coverage = np.mean(self.coverages[-500:])
                        mean_penalty = np.mean(self.penalties[-500:])
                        mean_steps = np.mean(self.steps[-500:])
                        mean_stops = np.mean(self.stopped[-500:])
                        print(f"[Episode {len(self.rewards)}] Reward: {mean_reward:.2f}, Coverage: {mean_coverage:.2f}, Penalty: {mean_penalty:.2f}, Steps: {mean_steps:.2f}, STOP: {mean_stops:.2f}")

        return True



class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check every 'check_freq' steps
        if self.n_calls % self.check_freq == 0:
            # Compute mean reward for the last 100 episodes
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Save the best model
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    if self.verbose > 0:
                        print(f"Best mean reward updated: {self.best_mean_reward:.2f}, model saved.")
        
        return True


def dataframe_to_ur_dict(df):
    return {col: set(df[col].dropna().unique()) for col in df.columns}

def write_run_description(save_path, description: str):
    with open(os.path.join(save_path, "description.txt"), "w") as f:
        f.write(description)


def moving_avg(data, window_size=50):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().to_numpy()

def plot_and_save_sns(metric, name, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    smooth_metric = moving_avg(metric)
    sns.lineplot(data=smooth_metric)
    plt.title(f"{name} over Episodes")
    plt.xlabel("Episode")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name.lower()}_curve.png"))
    plt.close()

def train_model(T, UR, sources, alpha, beta, gamma, save_path):
    value_index, source_stats = compute_UR_value_frequencies_in_sources(sources, UR)
    env = DataSelectionEnv(sources, UR, source_stats, value_index, alpha, beta, gamma)
    env.seed(SEED)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        buffer_size=10000,
        learning_rate=1e-3,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=500,
        tensorboard_log="./tb_logs",  # Optional, for debugging/plotting
        seed=42,  # For reproducibility
    )
    callback = MetricLoggerCallback()
    best_model_callback = SaveBestModelCallback(check_freq=1000, save_path=save_path)
    model.learn(total_timesteps=8000, callback=[callback, best_model_callback])


    os.makedirs(save_path, exist_ok=True)
    
    # Compose a description string for this run
    description = f"""Run description:
        Date: {datetime.datetime.now()}
        Alpha: {alpha}
        Beta: {beta}
        Gamma: {gamma}
        Seed: {SEED}
        Total timesteps: 8000
        User Request: {UR}
        Source Variant: {sources}
        Save path: {save_path}
        """

    write_run_description(save_path, description)
    try:
        shutil.copy("Coverage_Guided_Row_Selection/RL/RL_Env.py", os.path.join(save_path, "RL_Env.py"))
        shutil.copy("Coverage_Guided_Row_Selection/RL_TrainAll.py", os.path.join(save_path, "RL_TrainAll.py"))
    except Exception as e:
        print(f"[Warning] Could not copy source files: {e}")

    np.savez(os.path.join(save_path, "metrics.npz"),
             rewards=callback.rewards,
             coverages=callback.coverages,
             penalties=callback.penalties,
             steps=callback.steps,
             stopped=callback.stopped)

    try:
        np.savez(os.path.join(save_path, "qvalues.npz"),
                qvalues=np.array(callback.q_values, dtype=object))
        print(f"Q-values saved successfully to {os.path.join(save_path, 'qvalues.npz')}")
    except Exception as e:
        print(f"[Warning] Failed to save qvalues.npz: {e}")
        
        
    plot_and_save_sns(callback.rewards, "Reward", save_path)
    plot_and_save_sns(callback.coverages, "Coverage", save_path)
    plot_and_save_sns(callback.penalties, "Penalty", save_path)
    plot_and_save_sns(callback.steps, "Steps", save_path)
    plot_and_save_sns(callback.stopped, "STOP Action Used", save_path)

    model.save(os.path.join(save_path, "dqn_model"))

    # Return metrics to aggregate outside
    return {
        "rewards": callback.rewards,
        "coverages": callback.coverages,
        "penalties": callback.penalties,
        "steps": callback.steps,
        "stopped": callback.stopped
    }

def run_all():
    test_cases = TestCases()
    ur_cases = [26]  # Reduced for example, add more if needed
    source_variants = {
        "high_penalty": lambda ctor: ctor.high_penalty_sources(),
        "low_penalty": lambda ctor: ctor.low_penalty_sources(),
        "low_coverage": lambda ctor: ctor.low_coverage_sources(),
    }
    alpha_values = [0.6]
    beta_values = [0.3]

    all_metrics = {
        "rewards": {},
        "coverages": {},
        "penalties": {},
        "steps": {},
        "stopped": {}
    }

    for case_id in ur_cases:
        T, UR = test_cases.get_case(case_id)
        constructor = SourceConstructor(T, UR)

        for variant_name, variant_fn in source_variants.items():
            sources = variant_fn(constructor)

            for alpha in alpha_values:
                for beta in beta_values:
                    gamma = 1- alpha - beta  # Ensure gamma is derived from alpha and beta
                    save_dir = f"Report_resultsMovie_case26/case_{case_id}/{variant_name}/alpha_{alpha}_beta_{beta}"
                    print(f"Training: Case={case_id}, Source={variant_name}, Alpha={alpha}, Beta={beta}, Gamma={gamma}")
                    start_time = time.time()
                    metrics = train_model(T, UR, sources, alpha, beta, gamma, save_dir)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    print(f"Finished training for Case={case_id}, Source={variant_name}.Time: {elapsed:.2f} seconds.")
                    key = f"case{case_id}_{variant_name}_a{alpha}_b{beta}"
                    for metric_name in all_metrics.keys():
                        all_metrics[metric_name][key] = metrics[metric_name]

    # Save all collected metrics from all runs
    np.savez("all_training_metrics_Report_case26Only.npz", **all_metrics)
    print("All training metrics saved to all_training_metrics.npz")

if __name__ == "__main__":
    run_all()

