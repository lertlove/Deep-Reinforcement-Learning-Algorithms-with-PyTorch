import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
from environments.RateControl_Environment import RateControl_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.Passive_DQN import Passive_DQN
from agents.actor_critic_agents.Passive_SAC_Discrete import Passive_SAC_Discrete

config = Config()
config.seed = 1
config.num_episodes_to_run = 10000
config.ctu_width = 64
config.ctu_height = 64

config.experiment_name = "exp_7"
config.results_dir = f"/src/results/rc_results/{config.experiment_name}"
config.file_to_save_data_results = f"{config.results_dir}/rc_openImageNet_{config.num_episodes_to_run}_{config.seed}-{config.ctu_width}_{config.ctu_height}.pkl"
config.file_to_save_results_graph = f"{config.results_dir}/rc_openImageNet_graph-{config.num_episodes_to_run}_{config.seed}-{config.ctu_width}_{config.ctu_height}.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True


config.training_episode_per_eval = 20
config.trials = 50
config.use_ssd_insteadof_mse = True
config.save_and_load_meta_state = True
config.interval_save_result = 100
config.interval_save_policy = 100
config.model_dir = f"/src/results/rc_models/{config.experiment_name}"
config.file_to_save_policy = f"{config.model_dir}/Policy_{config.num_episodes_to_run}_{config.seed}-{config.ctu_width}_{config.ctu_height}.pt"
config.load_model_file = "/src/results/rc_models/exp_7/Policy_10000_1-64_64-Passive_SAC_Discrete-ep_4200-score_-802.84.pt"

config.environment = RateControl_Environment(config, 256)

config.hyperparameters = {
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
        "clip_rewards": False
    },
    "Actor_Critic_Agents":  {

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
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 512],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 1000,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__== '__main__':
    # AGENTS = [Passive_DQN]
    AGENTS = [Passive_SAC_Discrete]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


