import os
from os import path
import argparse
import cv2
import numpy as np
import image_slicer
from pathlib import Path
from math import log10, sqrt
# Importing Image module from PIL package 
from PIL import Image 
import PIL 


# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('-q','--quantization', nargs="+", type=int, dest='qpList', help='Quantization Levels', default=[10, 50, 100, 150, 200, 250])
# parser.add_argument('-f','--filename', type=str, dest='targetFile',  default="plane", help='Target Image File')
# args = parser.parse_args()

# qpList = args.qpList
# targetFile = args.targetFile
# dirName = targetFile
## creating a image object (main image) 
# image = Image.open(targetFile+".jpg") 

# try:
#     # Create target Directory
#     os.mkdir(dirName + "_split")
#     os.mkdir(dirName)
#     print("Directory " , dirName ,  " Created ") 
# except FileExistsError:
#     print("Directory " , dirName ,  " already exists")

def doQuantize(image,qp):
    # quantize a image 
    img = image.quantize(qp)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    fileName = "{}/{}_{}.jpg".format(dirName,targetFile,qp)
    img.save( fileName, 'jpeg')

    computePSNR(targetFile+".jpg",fileName)
    # to show specified image 
    img.show() 
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def computePSNR(originalFile,compressFile):
     original = cv2.imread(originalFile)
     compressed = cv2.imread(compressFile, 1)
     value = PSNR(original, compressed)
     print(f"{compressFile} - PSNR value is {value} dB")

def computeMSE(originalFile,compressFile):
    original = cv2.imread(originalFile)
    compressed = cv2.imread(compressFile, 1)
    mse = np.mean((original - compressed) ** 2)
    print(f"{compressFile} - MSE value is {mse}")
    return mse

def computeVariance(fileName):
    image = cv2.imread(fileName,0)
    (mean , stddv) = cv2.meanStdDev(image)
    # variance = np.var(image)
    # print(f"{fileName} - mean: {mean}, std:{stddv} => {stddv**2}, var:{variance}")
    return mean,stddv**2
    
def doQP():
    for qp in qpList:
        print(qp)
        # doQuantize(image,qp)
        fileName = "{}/{}_{}.jpg".format(dirName,targetFile,qp)
        # computeMSE(targetFile+".jpg",fileName)
        computeVariance(fileName)

def splitImageIntoTiles(filepath,x,y,dest,forceSplit=False):
    fileName = Path(filepath).stem
    image = cv2.imread(filepath)
    print(f"{filepath} shape:{image.shape}")
    splitFiles = []
    splitFolder = f"{dest}/{fileName}_split_{x}_{y}"
    if path.exists(splitFolder)==False or forceSplit:
        createTargetDir(dest)
        createTargetDir(splitFolder)
        for r in range(0,image.shape[0],x):
            for c in range(0,image.shape[1],y):
                saveFile = f"{splitFolder}/{fileName}_{r}_{c}.jpg"
                cv2.imwrite(saveFile,image[r:r+x, c:c+y,:])
                splitFiles.append(saveFile)
    else:
        for filename in os.listdir(splitFolder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                saveFile = os.path.join(splitFolder, filename)
                splitFiles.append(saveFile)
    return splitFiles

def doAllFilesInDir(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            # print(filepath)
            computeVariance(filepath)
        else:
            continue

def createTargetDir(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

# # splitImageIntoTiles(targetFile,620,530)
# directory = f'{dirName}_split'
# doAllFilesInDir(directory)
# computeVariance(targetFile+".jpg")