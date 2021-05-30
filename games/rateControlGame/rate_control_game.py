import os
import splitfolders
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../')))
# print(sys.path)
from utils import quant
from gameMode import GameMode
from pathlib import Path
import cv2

# DATASET_SPLIT_DIR='/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/dataset_split'
# DATASET_SOURCE_DIR = '/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/dataset'
# CTU_IMAGE_DIR = '/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/CTU_Images'
DATASET_SPLIT_DIR= abspath(join(dirname(__file__), '../../content/dataset_split'))
CTU_IMAGE_DIR = abspath(join(dirname(__file__), '../../content/CTU_Images'))
CTU_QUANTIZED_DIR = abspath(join(dirname(__file__), '../../content/CTU_quantized'))
# print(f"DATASET_SPLIT_DIR - {DATASET_SPLIT_DIR}")

CTU_WIDTH = 640
CTU_HEIGHT = 640
# TRAIN_MODE="train"
# TEST_MODE="test"
# GAME_MODE=TRAIN_MODE

# Simulate HM - HEVC Software
class RateControlGame():
    
    def __init__(self,image_dir,environment,reloadDataset=False):
        
        # print(f'image dir:{image_dir}')
        self.image_dir = image_dir
        self.episode_step = -1
        self.environment = environment
        self.mode = None
        self.imageFile = None
        self.ctuImages = []
        
        self.ctuVariants = []
        self.ctuShapes = []
        self.current_ctu = 0
        self.ctuSplitFolder = ""

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
        self.episode_step = self.episode_step + 1
        return self.doCompressCtu()
    
    def finishStep(self):
        self.current_ctu = self.current_ctu + 1
        self.doCompressCtu()

    def doCompressCtu(self):
        selectedQP = self.environment.request_action()
        filepath = self.ctuImages[self.current_ctu]
        # print(f"doCompressCtu self.current_ctu : {self.current_ctu}")
        # print(f"doCompressCtu filepath : {filepath}")
        destDir = f"{self.ctuSplitFolder}/{self.episode_step}"
        bitused, mse = quant.doQuantize(filepath,selectedQP,destDir)
        return bitused, mse

    def reset(self,mode):
        # game start
        print('Game reset')
        self.mode = mode
        self.current_ctu = 0
        # if mode==GameMode.TRAIN_MODE:
        imageData = self.fetch_image()
        targetBit = 1000 #1000kbps
        return targetBit, imageData
    
    # def next_episode(self):
    #     self.episode_step = self.episode_step + 1
    
    def fetch_image(self):
        imageFile = self.train[self.episode_step]
        fileName = Path(imageFile).stem
        self.ctuSplitFolder = f"{CTU_IMAGE_DIR}/{fileName}_split_{CTU_WIDTH}_{CTU_HEIGHT}"
        self.ctuImages = quant.splitImageIntoTiles(imageFile,CTU_WIDTH,CTU_HEIGHT,CTU_IMAGE_DIR)

        self.ctuVariants.clear()
        self.ctuShapes.clear()
        
        for ctuImage in self.ctuImages:
            ctu_height, ctu_width, variant = quant.computeVariance(ctuImage)
            # self.ctuMeans.append(mean)
            self.ctuVariants.append(variant)
            self.ctuShapes.append((ctu_height, ctu_width))

        frameImage = cv2.imread(imageFile)
        pic_height, pic_width, _ = frameImage.shape

        return pic_height, pic_width, len(self.ctuImages), self.ctuShapes, self.ctuVariants


    def getCurrentCtuShape():
        ctuImage = cv2.imread(ctuImages[self.current_ctu])
        ctu_height, ctu_width, _ = ctuImage.shape
        return ctu_height, ctu_width

# if __name__ == '__main__':
#     image_dir = DATASET_SOURCE_DIR
#     mode = GAME_MODE
#     # RateControlGame(image_dir).start(mode)
#     print("import rate control game")