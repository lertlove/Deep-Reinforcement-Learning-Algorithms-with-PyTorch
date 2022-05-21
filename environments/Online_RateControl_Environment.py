import copy

import random
import cv2
import gym
import numpy as np
import json
from gym import spaces
from gym.utils import seeding

from .experiments.experiment import Experiment

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../games')))

from gameMode import GameMode
from rateControlGame import RateControlGame
from utils import quant

NUM_QP_LEVELS = 52
MAX_NUM_CTUS = 100

GAME_MODE=GameMode.TRAIN_MODE

SOURCE_DIR = abspath(join(dirname(__file__), '../content/dataset'))

class Online_RateControl_Environment(gym.Env):
    environment_name = "Rate Control Environment"

    def __init__(self, config=None, environment_dimension=52, deterministic=False):
        
        self.config = config
        # self.game = RateControlGame(self,SOURCE_DIR)
        
        # actions list = number of qp levels 
        self.action_space = spaces.Discrete(NUM_QP_LEVELS)
        self.experiment = Experiment(config)
        
        # state space:
        # >Fixed value per episode
        # 1. Total Target Bit
        # 2. Total number of CTU
        # 3. Total Area (WxH)
        # >changing during observe
        # 4. percent bit balance
        # 5. percent of remaining CTU or percent of remaining area
        # 6. normalized variance of each CTU
        # 7. %of area
        # self.observation_space = spaces.Dict(dict(
        #     episodeVariables=spaces.Box(0, float('inf'), shape=(3,), dtype='float32'),
        #     observation=spaces.Box(0, 1, shape=(4,), dtype='float32'),
        # ))

        low = np.zeros(7)
        high = np.array([float('inf'), float('inf'), float('inf'), 1, 1, 1, 1])
        self.observation_space = spaces.Box(low,high)

        self.seed()
        self.reward_threshold = 0.0
        self.trials = self.config.trials
        # self.max_episode_steps = environment_dimension #number of ctus?
        self.id = "Online Rate Control"
        self.episode_step = -1
        self.environment_dimension = environment_dimension
        self.total_num_ctus = MAX_NUM_CTUS
        self.ctuImages = []
        
        self.ctuVariants = []
        self.ctuShapes = []
        self.current_ctu = 0
        self.currentBitUsed = 0
        self.currentMSE = 0
        # self.reward_for_achieving_goal = self.environment_dimension
        # self.step_reward_for_not_achieving_goal = -1

        self.deterministic = deterministic

        # agent delegate function for request action method
        print(f"onRequestAction=None")
        self.onStartExperiment = None
        self.onRequestAction = None
        self.onDoneAction = None
        self.onEndEpisode = None
        self.reset_game_agent = None
        print("init RateControl_Environment")
    
    def get_state_size(self):
        # self.episodeVariables = np.array([self.total_target_bit,self.total_num_ctus, self.total_area])

        # self.stateVariables = np.array([self.percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area, avg_remaining_ctu_variants, self.percent_remaining_area])

        # self.state = np.concatenate((self.episodeVariables, self.stateVariables))
        return 9

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_from_episode(self, episode):
        self.episode_step = episode

    # def start_game(self, agent_round):
    #     # do start game
    #     print("Environment - Do start game!")
    #     print("Game will then request action")
    #     self.currentBitUsed, self.currentMSE = self.game.start_game(agent_round)
    #     self.onDoneAction()

    def request_action(self, message, state=None):
        # Expect request from HM

        if state == None:
            state = self.state

        print(f"Environment - onRequestAction state : {state}")
        print(f"Environment - onRequestAction function : { self.onRequestAction}")
        action = self.onRequestAction(state, self.remaining_bit > 0)

        print(f"{self.current_ctu} - select action - {action}")
        assert action < NUM_QP_LEVELS , "You picked an invalid action"
        return action

    def initialize_episode(self, message):
        # start_espisode {'allNumPicCoded': 0, 'command': 'start_espisode', 'meta_data': 'video_filename', 'picHeight': 144, 'picWidth': 176, 'pocLast': 0, 'variants': [651.8190145492554, 267.4563970565796, 306.4253472222226, 123.95818996429443, 113.46095943450928, 53.200188530820014, 80.56809997558594, 97.61753845214844, 39.367078993058385]}

        print("Environment initialize_episode")
        self.step_count = 0
        self.current_ctu = 0
        self.currentBitUsed = 0
        self.currentMSE = 0

        self.episode_step = message["pocLast"]
        pic_height = message["picHeight"]
        pic_width = message["picWidth"]
        self.total_area = pic_width*pic_height
        
        self.ctuShapes = message["ctuAreas"]
        self.ctuVariants = message["variants"]
        self.total_num_ctus  = len(self.ctuVariants)
        self.maxVariant = max(self.ctuVariants)
    
    def setupTargetBits(self, message):
        # set target bit
        self.total_target_bit = message["estimatedBits"]
        self.initial_qp = message["sliceQP"] #may need to have initial qp as param
       
        self.remaining_bit = self.total_target_bit
        self.percent_bit_balance = 1

        self.reset_game_agent()
        # self.reset()
    
    def reset(self):

        percent_remaining_ctu = 1
        self.percent_remaining_area = 1

        # initial state - first ctu
        ctu_height, ctu_width = self.ctuShapes[self.current_ctu]
        ctu_variance = self.ctuVariants[self.current_ctu]/self.maxVariant if self.maxVariant != 0 else 0 #compute first ctu variance
        percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)

        avg_remaining_ctu_variants = np.mean(self.ctuVariants[self.current_ctu:self.total_num_ctus])/self.maxVariant if self.maxVariant != 0 else 0
        self.percent_remaining_area -= percent_ctu_area

        # should run initialize_episode & setupTargetBits before reset
        self.episodeVariables = np.array([self.total_target_bit,self.total_num_ctus, self.total_area])

        self.stateVariables = np.array([self.percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area, avg_remaining_ctu_variants, self.percent_remaining_area])

        self.state = np.concatenate((self.episodeVariables, self.stateVariables))
        self.next_state = None
        self.reward = None
        self.done = False
        print(f"reset: self.state = {self.state}")
        return self.state

    def done_action(self):
        # Expect response from HM
        # m_uiPicTotalBits += pCtu->getTotalBits();
        # m_dPicRdCost     += pCtu->getTotalCost();
        # m_uiPicDist      += pCtu->getTotalDistortion();
        
        # update from message?
        self.current_ctu = self.current_ctu + 1
        
        if self.config.reward_function == "RECIPROCAL":
            self.reward = 1/self.currentMSE if self.currentMSE >= 0.00001 else 100000 
            print(f"RECIPROCAL reward = {self.reward}")
        else:
            self.reward = -self.currentMSE #MINUS_MSE
            print(f"MINUS_MSE reward = {self.reward}")

        self.remaining_bit -= self.currentBitUsed
        self.percent_bit_balance = self.remaining_bit/self.total_target_bit

        if self.current_ctu < self.total_num_ctus:
            percent_remaining_ctu = (self.total_num_ctus-self.current_ctu)/self.total_num_ctus
            ctu_height, ctu_width = self.ctuShapes[self.current_ctu]
            ctu_variance = self.ctuVariants[self.current_ctu]/self.maxVariant if self.maxVariant != 0 else 0 #compute first ctu variance
            percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)
            avg_remaining_ctu_variants = np.mean(self.ctuVariants[self.current_ctu:self.total_num_ctus])/self.maxVariant if self.maxVariant != 0 else 0
            
            self.percent_remaining_area -= percent_ctu_area

        else:
            percent_remaining_ctu = 0
            ctu_variance = 0
            percent_ctu_area = 0
            avg_remaining_ctu_variants = 0
            self.percent_remaining_area = 0

        self.stateVariables = np.array([self.percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area, avg_remaining_ctu_variants, self.percent_remaining_area])
        self.next_state = np.concatenate((self.episodeVariables, self.stateVariables))

    def step(self, action):
        print(f'{self.current_ctu} - after action - {action}')
        self.step_count += 1
        self.done_action()
        if self.current_ctu >= self.total_num_ctus:
            self.done = True
        self.state = self.next_state
        print(f"self.episodeVariables = {self.episodeVariables}")
        print(f"self.state = {self.state}")
        return self.state, self.reward, self.done, {}

    def finishStep(self, message):
        # after_apply_qp = {'command': 'after_apply_qp', 'ctuBits': 3643, 'ctuCost': 219322.832756, 'ctuDist': 197890, 'ctuMSE': 48.31298828125, 'ctu_id': 0}
        
        self.current_ctu = message["ctu_id"]
        self.currentBitUsed = message["ctuBits"]
        # self.currentMSE = message["ctuDist"]
        self.currentMSE = message["ctuMSE"]
        
        ######################################## 
        self.onDoneAction() 
        # to perform 
        # 1) agent.step
        # 2) agent.conduct_action
        # 3) environment.step
        # 4) environment.done_action (update for next state)
        # 5) total_episode_score_so_far += self.reward 
        # 6) learn & save_experience()


# Command Function
    def start_experiment(self,message):
        # print(f"start_experiment {message}")
        # start_experiment {'GOPSize': 1, 'LCUHeight': 64, 'LCUWidth': 64, 'command': 'start_experiment', 'frameRate': 24, 'meta_data': 'video_filename', 'picHeight': 144, 'picWidth': 176, 'targetBitrate': 1000000, 'totalFrames': 5}

        self.onStartExperiment()
        reply = self.experiment.start_experiment(message)
        return reply
    
    def start_espisode(self,message):
        print(f"start_espisode {message}")
        # start_espisode {'allNumPicCoded': 0, 'command': 'start_espisode', 'meta_data': 'video_filename', 'picHeight': 144, 'picWidth': 176, 'pocLast': 0, 'variants': [651.8190145492554, 267.4563970565796, 306.4253472222226, 123.95818996429443, 113.46095943450928, 53.200188530820014, 80.56809997558594, 97.61753845214844, 39.367078993058385]}

        self.initialize_episode(message)

        reply = f"Environment: have got start_espisode = {message}"
        return reply
    
    def setup_targetBits(self, message):
        print(f"setup_targetBits {message}")
        # setup_targetBits {'command': 'setup_targetBits', 'estimatedBits': 41666, 'lambda': 6.417758, 'meta_data': 'video_filename', 'sliceQP': 22}
        
        self.setupTargetBits(message)

        reply = f"Environment: have got setup_targetBits = {message}"
        return reply

    def request_estimate_qp(self,message):
        print(f"request_estimate_qp {message}")
        # request_estimate_qp {'bitsCoded': 0, 'bitsLeft': 41666, 'command': 'request_estimate_qp', 'ctu_id': 0, 'estHeaderBits': 0, 'frameLevel': 0, 'lcuCoded': 0, 'lcuLeft': 9, 'lowerBound': 41666, 'numberOfLCU': 9, 'numberOfPixel': 25344, 'picEstQP': 22, 'pixelsLeft': 25344, 'targetBits': 41666}
        
        action = self.request_action(message)
        result = {"estimate_qp":action}

        reply = f"Environment: estimate_qp action = {json.dumps(result)}"
        print(reply)
        #return selected qp
        return json.dumps(result)

    def after_apply_qp(self,message):
        print(f"after_apply_qp {message}")
        # after_apply_qp {'command': 'after_apply_qp', 'ctuBits': 1559, 'ctuCost': 21056.855254000002, 'ctuDist': 7166, 'ctu_id': 7}
        self.finishStep(message)

        reply = f"Environment: have got after_apply_qp = {message}"
        return reply

    def end_episode(self,message):
        print(f"end_episode {message}")
        # end_episode {'command': 'end_episode', 'numPicCoded': 1, 'totalCoded': 5}
        
        self.onEndEpisode()
        reply = f"We have got end_episode = {message}"
        return reply