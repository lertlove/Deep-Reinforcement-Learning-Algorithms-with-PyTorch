import quant as q
from pathlib import Path
import cv2
import threading, queue
import sys

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

def test_show_image():
    filepath = "/mnt/nas/openImageNet/dataset/train/14702abb25310cee.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/1470dc1168584d9b.jpg"
    
    image = cv2.imread(filepath)
    print(f"image shape = {image.shape}")
    q.showImage("test",image)

def test_prepare_image():

    limit = sys.maxsize #30
    page_limit = 10000
    page = 0
    count = 0
    
    f = open(f"meta_{page}.txt","w+")
    lf = open(f"lf_meta_{page}.txt","w+")
    qq = queue.Queue()
    lfq = queue.Queue()

    def worker():
        
        nonlocal page_limit
        nonlocal page
        nonlocal count
        nonlocal f

        while True:
            item = qq.get()
            print(f'Working on {item}')
            f.write(item)
            count += 1
            if count >= page_limit:
                count = 0
                page +=1
                f.close()
                f = open(f"meta_{page}.txt","w+")
            qq.task_done()

    def lf_worker():
        nonlocal lf
        while True:
            item = lfq.get()
            print(f'Working on large file {item}')
            lf.write(item)
            lfq.task_done()

    def callback(filepath):

        print(filepath)
        image = cv2.imread(filepath)
        h,w,_ = image.shape
        isTooLarge = True if h*w/64/64 > 1000 else False
        # isTooLarge = True if w > 1000 else False
        item = f"{filepath},{h},{w},{isTooLarge}\n"
        
        if isTooLarge == True:
            lfq.put(item)
        else:
            qq.put(item)


    directory = "/mnt/nas/openImageNet/dataset/train/"
    # turn-on the worker thread
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=lf_worker, daemon=True).start()

    q.doLimitFilesInDir(directory,limit,callback)

    # block until all tasks are done
    qq.join()
    lfq.join()
    print('All work completed')
    if f.closed:
        print('file is closed')
    else:
        f.close()
    
    lf.close()

# ctuImages = test_split_to_tile()
# print(ctuImages)
# test_quantized()
# test_merge2()

# test_mse()
# test_show_image()
test_prepare_image()