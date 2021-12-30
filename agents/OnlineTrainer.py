import copy
import random
import pickle
import os
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from agents.Trainer import Trainer
from zeroMQ.ZmqServer import ZmqServer

class OnlineTrainer(Trainer):
    """Runs games for given agents. Optionally will visualise and save the results"""
    def __init__(self, config, agents):
        Trainer.__init__(self,config,agents)
        self.zmqServer = ZmqServer(config)
        self.zmqServer.start()
        print("OnlineTrainer end init")


    def run_games_for_agents(self):
        """Run a set of games for each agent. Optionally visualising and/or saving the results"""
        self.results = self.create_object_to_store_results()

        # for online training, should have only one agent
        for agent_number, agent_class in enumerate(self.agents):
            agent_name = agent_class.agent_name
            self.start_agent_server(agent_number + 1, agent_class)

        # Visualize Section
        #     if self.config.visualise_overall_agent_results:
        #         agent_rolling_score_results = [results[1] for results in  self.results[agent_name]]
        #         self.visualise_overall_agent_results(agent_rolling_score_results, agent_name, show_mean_and_std_range=True)
        # if self.config.file_to_save_data_results: self.save_obj(self.results, self.config.file_to_save_data_results)
        # if self.config.file_to_save_results_graph: plt.savefig(self.config.file_to_save_results_graph, bbox_inches="tight")
        # plt.show()
        # return self.results

    def start_agent_server(self, agent_number, agent_class):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_results = []
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_round = 1
        
        agent_config = copy.deepcopy(self.config)
        self.zmqServer.update_environment(agent_config.environment)

            # if self.environment_has_changeable_goals(agent_config.environment) and self.agent_cant_handle_changeable_goals_without_flattening(agent_name):
            #     print("Flattening changeable-goal environment for agent {}".format(agent_name))
            #     agent_config.environment = gym.wrappers.FlattenDictWrapper(agent_config.environment,
            #                                                                dict_keys=["observation", "desired_goal"])

        if self.config.randomise_random_seed: agent_config.seed = random.randint(0, 2**32 - 2)
        agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
        print("AGENT NAME: {}".format(agent_name))
        print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
        agent = agent_class(agent_config)
        self.environment_name = agent.environment_title
        print(agent.hyperparameters)

        print("RANDOM SEED: " , agent_config.seed)
        
        # summarize each episodes
        # game_scores, rolling_scores, time_taken = agent.run_n_episodes(agent_round=agent_round)
        # print("Time taken: {}".format(time_taken), flush=True)
        # self.print_two_empty_lines()
        # agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])
        # if self.config.visualise_individual_results:
        #     self.visualise_overall_agent_results([rolling_scores], agent_name, show_each_run=True)
                
        #         # if self.config.file_to_save_data_results: 
        #         #     results_path = os.path.splitext(self.config.file_to_save_data_results)[0]
        #         #     results_path = f"{results_path}-round_{agent_round}.pkl"
        #         #     self.save_obj(agent_results, results_path)
        #         # if self.config.file_to_save_results_graph:
        #         #     graph_path = os.path.splitext(self.config.file_to_save_results_graph)[0]
        #         #     graph_path = f"{graph_path}-round_{agent_round}.png"
        #         #     plt.savefig(graph_path, bbox_inches="tight")
        #         plt.show()
        #     agent_round += 1
        # self.results[agent_name] = agent_results








