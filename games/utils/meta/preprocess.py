import os
import sys
import re
import math
import pandas as pd
import shutil
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../../')))

from utils import quant

# INIT PARAMS
FPS = 2 #30

# time period in seconds
SHORT_TIME = 10 #20
MID_TIME = 60
LONG_TIME = 300
DEFAULT_TIME = SHORT_TIME

NPIC_CLIP = FPS*DEFAULT_TIME

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_PATH)

SOURCE_DIR = "test2"
SOURCE_DIR = os.path.join(CURRENT_PATH,SOURCE_DIR)

SPLIT_DIR = "split"
SPLIT_DIR = os.path.join(CURRENT_PATH,SPLIT_DIR)
quant.createTargetDir(SPLIT_DIR)

CSV_DIR = "csv_result"
CSV_DIR = os.path.join(CURRENT_PATH,CSV_DIR)
quant.createTargetDir(CSV_DIR)

METAFILE_BUFFER_SIZE = 20

def doSplitPicsInDir(sourcDir, targetDir, numDir):
    
    splitDirs = splitDirectory(targetDir, numTargetFolder)
    print(splitDirs)

    target_files = os.listdir(sourcDir)
    quant.sort_nicely(target_files)
    currentDir = 0
    runningNumber = 0

    metadata = []

    for filename in target_files:
        if filename.endswith(".txt"):
            filepath = os.path.join(sourcDir, filename)
            df = pd.read_csv(filepath,header=None)
            for index, row in df.iterrows():
                # print(f"{currentDir}:= {index}:{row[0]}")
                data = {}

                sourceFilePath = row[0]
                sourceFileID = Path(sourceFilePath).stem
                data["sourceFileID"] = sourceFileID
                data["sourceFilePath"] = sourceFilePath
                # print(f"sourceFilePath= {sourceFilePath}")
                # print(f"sourceFileID= {sourceFileID}")
                
                splitDir = splitDirs[currentDir]
                subID = f"{runningNumber:06d}"
                targetFileName = f'{subID}.jpg'
                targetFilePath = os.path.join(splitDir,targetFileName)
                data["targetFileID"] = f"{currentDir}-{subID}"
                data["targetFolderID"] = f"{currentDir}"
                data["subID"] = f"{subID}"
                data["targetFilePath"] = targetFilePath
                
                print(f"data:{data}")
                shutil.copy(sourceFilePath,targetFilePath)
                
                metadata.append(data)
                resultDF = pd.json_normalize(metadata)

                currentDir += 1
                currentDir %= numDir
                if currentDir%numDir == 0:
                    runningNumber += 1

                if len(metadata)%METAFILE_BUFFER_SIZE == 0:
                    print(resultDF.head())
                    resultDF.to_csv(f"{CSV_DIR}/meta_result_{runningNumber}.csv",index=False)
                    metadata = []
        else:
            continue
    
    if len(metadata) > 0:
        resultDF.to_csv(f"{CSV_DIR}/meta_result_{runningNumber}.csv",index=False)
        metadata = []


def findTotalPic(directory):
    
    target_files = os.listdir(directory)
    quant.sort_nicely(target_files)
    totalPic = 0
    for filename in target_files:
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            num_lines = sum(1 for line in open(filepath))
            totalPic += num_lines
            print(f"filepath:{filepath}, num_lines:{num_lines}")
            # computeVariance(filepath)
        else:
            continue
    return totalPic

def splitDirectory(prefixPath, numDir):
    splitDirs = []
    for i in range(0,numDir):
        splitDir = os.path.join(prefixPath,str(i))
        splitDirs.append(splitDir)
        quant.createTargetDir(splitDir)
    return splitDirs

# quant.createTargetDir(self.config.model_dir)

totalPic = findTotalPic(SOURCE_DIR)
print(f"totalPic = {totalPic}")
print(f"NPIC_CLIP = {NPIC_CLIP}")

numTargetFolder = math.ceil(totalPic/NPIC_CLIP)
print(f"numTargetFolder = {numTargetFolder}")

doSplitPicsInDir(SOURCE_DIR, SPLIT_DIR, numTargetFolder)
