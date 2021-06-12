import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
from environments.RateControl_Environment import RateControl_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.Passive_DQN import Passive_DQN

config = Config()
config.seed = 1
config.environment = RateControl_Environment(256)
config.num_episodes_to_run = 1000
config.file_to_save_data_results = "rc_models/rc_openImageNet_1000_3-64_64.pkl"
config.file_to_save_results_graph = "rc_models/rc_openImageNet_graph-1000_3-64_64.png"
config.show_solution_score = False
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True


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
    }
}

if __name__== '__main__':
    AGENTS = [Passive_DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


