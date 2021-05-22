import os
import splitfolders
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../')))
# print(sys.path)
from utils import quant
from gameMode import GameMode


# DATASET_SPLIT_DIR='/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/dataset_split'
# DATASET_SOURCE_DIR = '/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/dataset'
# CTU_IMAGE_DIR = '/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/CTU_Images'
DATASET_SPLIT_DIR= abspath(join(dirname(__file__), '../../content/dataset_split'))
CTU_IMAGE_DIR = abspath(join(dirname(__file__), '../../content/CTU_Images'))
# print(f"DATASET_SPLIT_DIR - {DATASET_SPLIT_DIR}")

CTU_WIDTH = 640
CTU_HEIGHT = 640
# TRAIN_MODE="train"
# TEST_MODE="test"
# GAME_MODE=TRAIN_MODE

class RateControlGame():
    
    def __init__(self,image_dir,environment,reloadDataset=False):
        
        # print(f'image dir:{image_dir}')
        self.image_dir = image_dir
        self.episode_step = 0
        self.environment = environment
        self.mode = None

        # prepare dataset
        if reloadDataset==True:
            splitfolders.ratio(self.image_dir, output=DATASET_SPLIT_DIR, seed=1337, ratio=(.8, 0.1,0.1))
        
        self.train = []
        train_dir= DATASET_SPLIT_DIR + "/" + GameMode.TRAIN_MODE.value
        print(f"train_dir : {train_dir}")

        for (root,dirs,files) in os.walk(train_dir, topdown=True):
            for name in files:
                filepath = os.path.join(root, name)
                # print(filepath)
                self.train.append(filepath)
        
        # print(f"train data : {self.train}")
        print(f"train data")

    def start_game(self):
        print("Game - Do start game!")
        self.environment.request_action()


    def reset(self,mode):
        # game start
        print('start')
        self.mode = mode
        # if mode==GameMode.TRAIN_MODE:
        imageData = self.fetch_image()
        targetBit = 1000 #1000kbps
        return targetBit, imageData
    
    def next_episode(self):
        self.episode_step = self.episode_step + 1
        self.environment.endEpisode()
    
    def fetch_image(self):
        imageFile = self.train[self.episode_step]
        ctuImages = quant.splitImageIntoTiles(imageFile,CTU_WIDTH,CTU_HEIGHT,CTU_IMAGE_DIR)
        ctuMeans = []
        ctuVariants = []
        
        for ctuImage in ctuImages:
            mean, variant = quant.computeVariance(ctuImage)
            ctuMeans.append(mean)
            ctuVariants.append(variant)

        return imageFile, ctuImages, ctuMeans, ctuVariants
        # self.environment.onCTUSplitDone(ctuImages)


# if __name__ == '__main__':
#     image_dir = DATASET_SOURCE_DIR
#     mode = GAME_MODE
#     # RateControlGame(image_dir).start(mode)
#     print("import rate control game")