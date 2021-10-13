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

class Online_DQN(DQN):
    """A deep Q learning agent"""
    agent_name = "Online_DQN"
    def __init__(self, config):
        print(f"Online_DQN init")
        DQN.__init__(self, config)
        self.passive = True
        # delegate function for request action method
        # print(f"Online_DQN environment: {hex(id(self.environment))}")
        self.environment.onStartExperiment = self.reset_game
        self.environment.onRequestAction = self.pick_action
        print(f"environment.onRequestAction init: {self.environment.onRequestAction}")
        self.environment.onDoneAction = self.step
        
        
    def get_state_size(self):
        print(f"Online_DQN get_state_size")
        return self.environment.get_state_size()

    # def reset_game(self):
    #     super(Passive_DQN, self).reset_game()
    #     self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def pick_action(self, state=None, isRemaining=True):
        if isRemaining:
            self.action = super().pick_action(state)
        else:
            self.action = 0
        print(f"Agent - pick action - {self.action}")
        return self.action

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
            # no need finish step in online mode
            # since it waits for zmq message trigger event
            # self.environment.finishStep()
            print(f"Agent - done = {self.done} finish step")
        else:
            print(f"Agent - done = {self.done} finish episode")
            self.episode_number += 1

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far/self.environment.total_area)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        
        if self.config.interval_save_result is not None:
            if self.episode_number%self.config.interval_save_result == 0 and self.config.file_to_save_data_results: 
                self.save_result_to_file()
        
        self.save_max_result_seen()
    
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.seed)
        self.state = None #self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))
