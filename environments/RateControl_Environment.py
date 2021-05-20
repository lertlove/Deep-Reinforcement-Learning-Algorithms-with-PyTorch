import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

NUM_QP_LEVELS = 256
MAX_NUM_CTUS = 100

class RateControl_Environment(gym.Env):
    environment_name = "Rate Control Environment"

    def __init__(self, environment_dimension=256, deterministic=False):

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
        # self.reward_for_achieving_goal = self.environment_dimension
        # self.step_reward_for_not_achieving_goal = -1

        self.deterministic = deterministic

        # agent delegate function for request action method
        self.onRequestAction = None
        self.onDoneAction = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start(self):
        # do start game
        print("Do start game!")
        print("Game will then request action")

    def request_action_observer(self,state=None):
        # Expect request from HM
        action = self.onRequestAction(state)
        print(f'{self.current_ctu} - select action - {action}')
        assert action > NUM_QP_LEVELS , "You picked an invalid action"
        return action

    def reset(self):
        # retrieve new image
        # Reset the state of the environment to an initial state
        self.total_target_bit = INITIAL_TARGET_BIT #or random target bit, but fix image
        self.total_num_ctus = MAX_NUM_CTUS #number of ctu tiles
        self.total_area = pic_width*pic_height
        self.episodeVariables = np.array(self.total_target_bit,self.total_num_ctus,self.total_area)
        
        percent_bit_balance = 1
        percent_remaining_ctu = 1
        ctu_variance = 0 #compute first ctu variance
        percent_ctu_area = (ctu_width*ctu_height)/(self.total_area)
        self.state = percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area
        
        self.current_ctu = 0
        self.next_state = None
        self.reward = None
        self.done = False
        # self.episode_steps = 0

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
        self.next_state = percent_bit_balance, percent_remaining_ctu, ctu_variance, percent_ctu_area
        self.reward = -distortion
        self.onDoneAction()
        
        return True

    def step(self, action):
        print(f'{self.current_ctu} - after action - {action}')
        self.step_count += 1
        if self.current_ctu > self.total_num_ctus:
            self.done = True
        self.state = self.next_state

        return {"observation": np.array(self.state), "episodeVariables": self.episodeVariables}, self.reward, self.done, {}