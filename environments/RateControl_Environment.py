import copy

import random
import cv2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../games')))

from gameMode import GameMode
from rateControlGame import RateControlGame
from utils import quant

NUM_QP_LEVELS = 256
MAX_NUM_CTUS = 100

GAME_MODE=GameMode.TRAIN_MODE

SOURCE_DIR = abspath(join(dirname(__file__), '../content/dataset'))

class RateControl_Environment(gym.Env):
    environment_name = "Rate Control Environment"

    def __init__(self, config=None, environment_dimension=256, deterministic=False):
        
        self.config = config
        self.game = RateControlGame(self,SOURCE_DIR)
        
        # actions list = number of qp levels 
        self.action_space = spaces.Discrete(NUM_QP_LEVELS)
        
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
        self.trials = 50
        self.max_episode_steps = environment_dimension #number of ctus?
        self.id = "Rate Control"
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
        self.onRequestAction = None
        self.onDoneAction = None
        print("init RateControl_Environment")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_from_episode(self, episode):
        self.game.start_from_episode(episode)

    def start_game(self, agent_round):
        # do start game
        print("Environment - Do start game!")
        print("Game will then request action")
        self.currentBitUsed, self.currentMSE = self.game.start_game(agent_round)
        self.onDoneAction()

    def request_action(self,state=None):
        # Expect request from HM

        if state == None:
            state = self.state

        print(f"Environment - onRequestAction state : {state}")
        if self.remaining_bit > 0:
            action = self.onRequestAction(state)
        else:
            action = 0 #lowest qp for jpeg encoding (qp =action +1)

        print(f"{self.current_ctu} - select action - {action}")
        assert action < NUM_QP_LEVELS , "You picked an invalid action"
        return action

    def reset(self):
        # retrieve new image
        print("Environment reset")
        self.step_count = 0
        self.current_ctu = 0
        self.currentBitUsed = 0
        self.currentMSE = 0
        targetBit, imageData = self.game.reset(GAME_MODE)
        pic_height, pic_width, total_num_ctus, self.ctuShapes, self.ctuVariants, filesize = imageData

        # print(f"ctuMeans : {ctuMeans}")
        # print(f"ctuVariants : {self.ctuVariants}")
        # Reset the state of the environment to an initial state
        self.maxVariant = max(self.ctuVariants)
        self.total_target_bit = targetBit #INITIAL_TARGET_BIT #or random target bit, but fix image
        self.total_num_ctus = total_num_ctus #MAX_NUM_CTUS #number of ctu tiles
        
        self.total_area = pic_width*pic_height
        self.episodeVariables = np.array([self.total_target_bit,self.total_num_ctus,self.total_area])
        
        self.remaining_bit = self.total_target_bit
        self.percent_bit_balance = 1
        percent_remaining_ctu = 1
        self.percent_remaining_area = 1

        ctu_height, ctu_width = self.ctuShapes[self.current_ctu]
        ctu_variance = self.ctuVariants[self.current_ctu]/self.maxVariant #compute first ctu variance
        percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)

        avg_remaining_ctu_variants = np.mean(self.ctuVariants[self.current_ctu:self.total_num_ctus])/self.maxVariant
        self.percent_remaining_area -= percent_ctu_area
        
        self.stateVariables = np.array([self.percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area, avg_remaining_ctu_variants, self.percent_remaining_area])
        
        self.state = np.concatenate((self.episodeVariables, self.stateVariables))
        self.next_state = None
        self.reward = None
        self.done = False

        return self.state

    def done_action(self):
        # Expect response from HM
        # m_uiPicTotalBits += pCtu->getTotalBits();
        # m_dPicRdCost     += pCtu->getTotalCost();
        # m_uiPicDist      += pCtu->getTotalDistortion();
        self.current_ctu = self.current_ctu + 1
        self.reward = 1/self.currentMSE if self.currentMSE >= 0.001 else 1000 

        self.remaining_bit -= self.currentBitUsed
        self.percent_bit_balance = self.remaining_bit/self.total_target_bit

        if self.current_ctu < self.total_num_ctus:
            percent_remaining_ctu = (self.total_num_ctus-self.current_ctu)/self.total_num_ctus
            ctu_height, ctu_width = self.ctuShapes[self.current_ctu]
            ctu_variance = self.ctuVariants[self.current_ctu]/self.maxVariant #compute first ctu variance
            percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)
            avg_remaining_ctu_variants = np.mean(self.ctuVariants[self.current_ctu:self.total_num_ctus])/self.maxVariant
            
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
        if self.current_ctu > self.total_num_ctus:
            self.done = True
        self.state = self.next_state
        print(f"self.episodeVariables = {self.episodeVariables}")
        print(f"self.state = {self.state}")
        return self.state, self.reward, self.done, {}

    def finishStep(self):
        # do next compress
        
        self.currentBitUsed, self.currentMSE = self.game.finishStep()
        self.onDoneAction()


# if __name__ == '__main__':
    # image_dir = SOURCE_DIR
    # environment = RateControl_Environment()
#     environment.reset()
#     environment.start_game()
#     print("Run game environment >>>")
