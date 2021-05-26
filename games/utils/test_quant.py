import quant as q

def test_split_to_tile():
    filepath = '../../content/dataset/plane/plane.jpg'
    destDir = '../../content/CTU_Images'
    q.splitImageIntoTiles(filepath,640,640,destDir)

def test_quantized():
    filepath = '../../content/CTU_Images/c56ee63069374df9_split_640_640/c56ee63069374df9_0_0.jpg'
    destDir = '../../content/CTU_Images/c56ee63069374df9_split_640_640/1'
    q.doQuantize(filepath,50,destDir)

test_split_to_tile()
# test_quantized()