import quant as q
from pathlib import Path

def test_split_to_tile():

    CTU_WIDTH = 64
    CTU_HEIGHT = 64
    CTU_IMAGE_DIR = '../../content/CTU_Images'
    filepath = '../../content/dataset_split/train/airplane/0a9d17fa73941bac.jpg'
    fileName = Path(filepath).stem
    
    return q.splitImageIntoTiles(filepath,CTU_WIDTH,CTU_HEIGHT,CTU_IMAGE_DIR)

def test_quantized():
    filepath = '../../content/CTU_Images/c56ee63069374df9_split_640_640/c56ee63069374df9_0_0.jpg'
    destDir = '../../content/CTU_Images/c56ee63069374df9_split_640_640/1'
    q.doQuantize(filepath,50,destDir)

def test_merge():
    originalFile = "/src/content/dataset_split/train/hammer/a4aa1e0a3af42b26.jpg"
    targetDir = "/src/content/CTU_Images/a4aa1e0a3af42b26_split_640_640/1_9"
    mergeImage = q.mergeImages(originalFile,targetDir)

def test_merge2():
    originalFile = "/src/content/dataset_split/train/hammer/a4aa1e0a3af42b26.jpg"
    targetDir = "/src/content/CTU_Images/a4aa1e0a3af42b26_split_640_640"
    mergeImage = q.mergeImages(originalFile,targetDir)

def test_mse():
    originalFile = "/src/content/dataset_split/train/hammer/a4aa1e0a3af42b26.jpg"
    targetFile = "/src/content/CTU_Images/a4aa1e0a3af42b26_split_640_640/merged/merged_CTU_Images-a4aa1e0a3af42b26_split_640_640.jpg"
    mse = q.computeMSEFromFiles(originalFile,targetFile)
    
# ctuImages = test_split_to_tile()
# print(ctuImages)
# test_quantized()
# test_merge2()

test_mse()