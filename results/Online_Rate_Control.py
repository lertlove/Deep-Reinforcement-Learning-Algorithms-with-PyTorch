import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

import gym
from environments.RateControl_Environment import RateControl_Environment
from environments.Online_RateControl_Environment import (
    Online_RateControl_Environment as Online_RateControl_Environment,
)
from environments.Online_RateControl_LimitQP_Environment import (
    Online_RateControl_Environment as Online_RateControl_Environment_LimitQP,
)
from agents.Trainer import Trainer
from agents.OnlineTrainer import OnlineTrainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.Passive_DQN import Passive_DQN
from agents.DQN_agents.Online_DQN import Online_DQN
from agents.actor_critic_agents.Passive_SAC_Discrete import Passive_SAC_Discrete

config = Config()
config.seed = 1
config.num_episodes_to_run = 20000
config.ctu_width = 16
config.ctu_height = 16

config.experiment_name = "exp_2-14"
config.results_dir = f"/src/results/rc_results/{config.experiment_name}"
config.file_to_save_data_results = (
    f"{config.results_dir}/rc_hevc_{config.experiment_name}_{config.seed}.pkl"
)
config.file_to_save_results_graph = (
    f"{config.results_dir}/rc_hevc_graph-{config.experiment_name}_{config.seed}.png"
)
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True
# config.reward_function = "RECIPROCAL"

config.training_episode_per_eval = 5
config.trials = 10
config.use_ssd_insteadof_mse = True
config.save_and_load_meta_state = True
config.model_dir = f"/src/results/rc_models/{config.experiment_name}"
config.file_to_save_policy = f"{config.model_dir}/Policy_{config.num_episodes_to_run}_{config.seed}-{config.ctu_width}_{config.ctu_height}.pt"
# config.load_model_file = "/src/results/rc_models/exp_1-3/Policy_20000_1-16_16-Online_DQN-ep_3000-score_-0.43281.pt"


config.trainMode = True
if config.trainMode:
    config.interval_save_result = 20
    config.interval_save_policy = 20
else:
    config.interval_save_result = None
    config.interval_save_policy = None

config.LimitQP = True
if config.LimitQP:
    config.environment = Online_RateControl_Environment_LimitQP(config, 64)
else:
    config.environment = Online_RateControl_Environment(config, 64)

config.hyperparameters = {
    "Online_DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 16,
        "buffer_size": 256,
        "epsilon_decay_rate_denominator": 150,
        "discount_rate": 0.999,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [300, 400],
        "final_layer_activation": None,
        "batch_norm": True,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "learning_iterations": 1,
        "clip_rewards": False,
    },
    "Passive_DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "buffer_size": 20,
        "epsilon_decay_rate_denominator": 150,
        "discount_rate": 0.999,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [400, 300],
        "final_layer_activation": None,
        "y_range": (-1, 14),
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "learning_iterations": 1,
        "clip_rewards": False,
    },
    "Actor_Critic_Agents": {
        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 512],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier",
        },
        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 512],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier",
        },
        "min_steps_before_learning": 1000,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
    },
}

if __name__ == "__main__":
    # AGENTS = [Passive_DQN]
    AGENTS = [Online_DQN]
    # trainer = Trainer(config, AGENTS)
    # trainer.run_games_for_agents()
    trainer = OnlineTrainer(config, AGENTS)
    trainer.run_games_for_agents()
