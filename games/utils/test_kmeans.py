import os
import torch_kmeans as km
from pathlib import Path
import cv2
import threading, queue
import sys


def test_quantized():
    filepath = '../../content/CTU_Images/c56ee63069374df9_split_640_640/c56ee63069374df9_0_0.jpg'
    destDir = '../../content/CTU_Images/c56ee63069374df9_split_640_640/1'
    q.doQuantize(filepath,50,destDir)

def test_show_image():
    filepath = "/mnt/nas/openImageNet/dataset/train/14702abb25310cee.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/1470dc1168584d9b.jpg"
    filepath =  "/mnt/nas/openImageNet/dataset/train/1ebb35691b23f11e.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/4728d246e8333a50.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/d279808768538a7b.jpg"
    image = cv2.imread(filepath)
    print(f"image shape = {image.shape}")
    cv2.imshow("test",image)
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 

def test_resizeImage():
    filepath =  "/mnt/nas/openImageNet/dataset/train/1ebb35691b23f11e.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/4728d246e8333a50.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/d279808768538a7b.jpg"
    km.resizeImage(filepath)

def test_doCompress():
    filepath =  "/mnt/nas/openImageNet/dataset/train/1ebb35691b23f11e.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/4728d246e8333a50.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/d279808768538a7b.jpg"
    km.doCompress(filepath)

# test_show_image()
# test_resizeImage()
test_doCompress()