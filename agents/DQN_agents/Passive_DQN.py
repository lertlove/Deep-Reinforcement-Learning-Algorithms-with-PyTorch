from agents.DQN_agents.DQN import DQN
from collections import Counter

import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer

class Passive_DQN(DQN):
    """A deep Q learning agent"""
    agent_name = "Passive_DQN"
    def __init__(self, config):
        DQN.__init__(self, config)
        self.passive = True
        # delegate function for request action method
        self.environment.onRequestAction = self.pick_action
        self.environment.onDoneAction = self.step

    # def reset_game(self):
    #     super(Passive_DQN, self).reset_game()
    #     self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        if not self.done:
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
            self.environment.finishStep()
            print(f"Agent - done = {self.done} finish step")
        else:
            print(f"Agent - done = {self.done} finish episode")
            self.episode_number += 1
    
    def start(self):
        """Passive Agent will trigger start signal to the game environment, and wait for the game response to train the nn model online"""
        self.environment.start()
    