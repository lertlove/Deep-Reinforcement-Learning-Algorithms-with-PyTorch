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

# ctuImages = test_split_to_tile()
# print(ctuImages)
# test_quantized()