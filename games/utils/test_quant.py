import quant as q

def test_split_to_tile():
    filename = '../../content/dataset/plane/plane.jpg'
    destDir = '../../content/CTU_Images'
    q.splitImageIntoTiles(filename,640,640,destDir)

test_split_to_tile()
