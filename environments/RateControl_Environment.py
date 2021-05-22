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
SOURCE_DIR = "/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/dataset"

class RateControl_Environment(gym.Env):
    environment_name = "Rate Control Environment"

    def __init__(self, environment_dimension=256, deterministic=False):
        
        self.game = RateControlGame(SOURCE_DIR,self)
        
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
        self.observation_space = spaces.Dict(dict(
            episodeVariables=spaces.Box(0, float('inf'), shape=(3,), dtype='float32'),
            observation=spaces.Box(0, 1, shape=(4,), dtype='float32'),
        ))

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 50
        self.max_episode_steps = environment_dimension #number of ctus?
        self.id = "Rate Control"
        self.environment_dimension = environment_dimension
        self.total_num_ctus = MAX_NUM_CTUS
        self.ctuImages = []
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

    def start_game(self):
        # do start game
        print("Environment - Do start game!")
        print("Game will then request action")
        return self.game.start_game()

    def request_action(self,state=None):
        # Expect request from HM
        print("Environment - onRequestAction")
        action = self.onRequestAction(state)
        print(f"{self.current_ctu} - select action - {action}")
        assert action > NUM_QP_LEVELS , "You picked an invalid action"
        return action

    def reset(self):
        # retrieve new image
        print("reset")
        targetBit, imageData = self.game.reset(GAME_MODE)
        imageFile, ctuImages, ctuMeans, ctuVariants = imageData
        # print(f"ctuMeans : {ctuMeans}")
        # print(f"ctuVariants : {ctuVariants}")
        # Reset the state of the environment to an initial state
        self.total_target_bit = targetBit #INITIAL_TARGET_BIT #or random target bit, but fix image
        self.total_num_ctus = len(ctuImages) #MAX_NUM_CTUS #number of ctu tiles
        # Image
        frameImage = cv2.imread(imageFile)
        pic_height, pic_width, _ = frameImage.shape

        self.total_area = pic_width*pic_height
        self.episodeVariables = np.array([self.total_target_bit,self.total_num_ctus,self.total_area])
        self.current_ctu = 0
        ctuImage = cv2.imread(ctuImages[self.current_ctu])
        ctu_height, ctu_width, _ = ctuImage.shape
        
        percent_bit_balance = 1
        percent_remaining_ctu = 1
        ctu_variance = 0 #compute first ctu variance
        percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)
        self.state = [percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area]
        
        self.next_state = None
        self.reward = None
        self.done = False

        return {"observation": np.array(self.state), "episodeVariables": self.episodeVariables}

    def done_action_observer(self,next_ctu,remaining_bit, norm_variance, ctu_width, ctu_height, distortion):
        # Expect response from HM
        # m_uiPicTotalBits += pCtu->getTotalBits();
        # m_dPicRdCost     += pCtu->getTotalCost();
        # m_uiPicDist      += pCtu->getTotalDistortion();
        percent_bit_balance = remaining_bit/self.total_target_bit
        percent_remaining_ctu = (self.total_num_ctus-next_ctu-1)/self.total_num_ctus
        ctu_variance = norm_variance
        percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)

        self.current_ctu = next_ctu
        self.next_state = [percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area]
        self.reward = -distortion
        self.onDoneAction()
        
        return True

    def step(self, action):
        # print(f'{self.current_ctu} - after action - {action}')
        self.step_count += 1
        if self.current_ctu > self.total_num_ctus:
            self.done = True
        self.state = self.next_state

        return {"observation": np.array(self.state), "episodeVariables": self.episodeVariables}, self.reward, self.done, {}

    def onCTUSplitDone(self,ctuImages):
        self.ctuImages = ctuImages
        print(self.ctuImages)

if __name__ == '__main__':
    image_dir = SOURCE_DIR
    environment = RateControl_Environment()
    environment.reset()
    environment.start_game()
    print("Run game environment >>>")
