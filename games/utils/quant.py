import os
from os import path
import re
import argparse
import cv2
import numpy as np
# import image_slicer
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

def doQuantize(filepath, qp, dest, useSSD=False):

    fileName = Path(filepath).stem
    image = Image.open(filepath) 
    
    quantizedDir = f"{dest}"
    createTargetDir(quantizedDir)

    # quantize a image 
    img = image.quantize(qp)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    quantizedFile = "{}/{}-qp_{}.jpg".format(quantizedDir,fileName,qp)
    img.save(quantizedFile, 'jpeg')
    filesize = os.path.getsize(quantizedFile)*8 # in bit usage

    mse = computeMSEFromFiles(filepath, quantizedFile, useSSD)

    return quantizedFile, filesize, mse
    # # to show specified image 
    # img.show() 
  
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

def computeMSEFromFiles(originalFile,compressFile,useSSD=False):
    original = cv2.imread(originalFile)
    compressed = cv2.imread(compressFile, 1)
    
    return computeMSE(original,compressed,useSSD)

def computeMSE(original,compressed,useSSD=False):

    err = np.sum((original.astype ("float") - compressed.astype ("float")) ** 2)
    if useSSD:
        print(f"SSD value is {err}")
        return err
    else:
        err/= float (original.shape [0] * original.shape [1])
        print(f"MSE value is {err}")
        return err

def computeVariance(fileName):
    image = cv2.imread(fileName,0)
    ctu_height, ctu_width = image.shape
    # (mean , stddv) = cv2.meanStdDev(image)
    variance = np.var(image)
    # print(f"{fileName} - mean: {mean}, std:{stddv} => {stddv**2}, var:{variance}")
    return ctu_height, ctu_width, variance
    
def doQP():
    for qp in qpList:
        print(qp)
        # doQuantize(image,qp)
        # fileName = "{}/{}_{}.jpg".format(dirName,targetFile,qp)
        # computeMSEFromFiles(targetFile+".jpg",fileName)
        # computeVariance(fileName)

def splitImageIntoTiles(filepath, x, y, dest, forceSplit=False):
    fileName = Path(filepath).stem
    image = cv2.imread(filepath)
    print(f"{filepath} shape:{image.shape}")
    splitFiles = []
    splitFolder = f"{dest}/{fileName}_split_{x}_{y}"
    if path.exists(splitFolder)==False or forceSplit:
        createTargetDir(dest)
        createTargetDir(splitFolder)
        ctu_index = 0
        for r in range(0,image.shape[0],x):
            for c in range(0,image.shape[1],y):
                saveFile = f"{splitFolder}/{ctu_index}-{fileName}-{r}_{c}.jpg"
                cv2.imwrite(saveFile,image[r:r+x, c:c+y,:])
                splitFiles.append(saveFile)
                ctu_index = ctu_index + 1
    else:
        for filename in os.listdir(splitFolder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                saveFile = os.path.join(splitFolder, filename)
                splitFiles.append(saveFile)
    
    sort_nicely(splitFiles)

    return splitFiles



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    fileName = Path(s).stem
    return [ tryint(c) for c in re.split('([0-9]+)', fileName) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def doAllFilesInDir(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            # print(filepath)
            # computeVariance(filepath)
        else:
            continue

def doLimitFilesInDir(directory,limit,_callback):
    fileCount = 0
    for (root,dirs,files) in os.walk(directory, topdown=True):
        for name in files:
            fileCount += 1
            if fileCount <= limit:
                filepath = os.path.join(root, name)
                # do something
                _callback(filepath)
            else:
                print(f"Load {fileCount-1} files")
                return

def createTargetDir(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

def mergeImages(originalFile,targetDir):
    print("merge images")
    orgImage = cv2.imread(originalFile)
    # print(f"orgImage.shape = {orgImage.shape}")
    mergeImage = np.zeros(orgImage.shape)
    # print(f"mergeImage.shape = {mergeImage.shape}")
    ctuFiles = os.listdir(targetDir)
    sort_nicely(ctuFiles)
    for filePath in ctuFiles:
        if filePath.endswith(".jpg") or filePath.endswith(".png"):
            filename = Path(filePath).stem
            # metaData = filename.split("-")
            ctu_index, filename, yx, qp = filename.split("-")
            y, x = map(int, yx.split("_"))
            qp = qp.split("_")[1]
            filePath = os.path.join(targetDir, filePath)
            print(f"filePath {filePath}")
            ctuImg = cv2.imread(filePath)
            h,w,_ = ctuImg.shape
            # showImage("ctuImg",ctuImg)

            # ctu_index, filename, yx = filename.split("-")
            # y, x = map(int, yx.split("_"))
            
            # filePath = os.path.join(targetDir, filePath)
            # print(f"filePath {filePath}")
            # ctuImg = cv2.imread(filePath)
            # h,w,_ = ctuImg.shape

            mergeImage[y:y+h, x:x+w,:] = ctuImg

    # mergeImage /= 255
    parentDir,tail = os.path.split(targetDir)
    mergePrefix = os.path.split(parentDir)[1]
    mergeDir = f"{parentDir}/merged"
    createTargetDir(mergeDir)
    mergePath = f"{mergeDir}/merged_{mergePrefix}-{tail}.jpg"

    ctuImg = cv2.imwrite(mergePath,mergeImage)
    # showImage("mergeImage",mergeImage)
    return mergeImage


def showImage(windowName,image):
    
    cv2.imshow(windowName,image)
    cv2.waitKey(0) 
    
    cv2.destroyAllWindows()