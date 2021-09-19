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
    filepath = "/mnt/nas/openImageNet/dataset/train/bcab5d9f876e42e9.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/a15bd01a96ca9317.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/057caf68d7af76b1.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/75464ee6a56a399d.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/7e66c0e6ee2b9be2.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/c5837d05d9265c54.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/59fe2bbd288c9fdd.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/04fb5d356acc7e77.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/0ececd39da631d22.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/25954f3e51b7f233.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/023bfb7176127a64.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/cbb9281a62c32b1b.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/dc9f547c27a2c230.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/d77c69397709c4b8.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/0cc19d050f6364ac.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/7086b8748c59af23.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/97ffa28ebd1d4dec.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/0033fa87c5bc7cba.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/f5a7473e406e9678.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/7e09be6afac5cfa6.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/88afa4fb20f9dc81.jpg"
    
    # filepath = "/mnt/nas/openImageNet/dataset/train/f265424baf3c98af.jpg"
    # filepath = "/mnt/nas/openImageNet/dataset/train/310388281f26d6d5.jpg"
    # filepath = "/mnt/nas/openImageNet/dataset/train/4728d246e8333a50.jpg"
    # filepath = "/mnt/nas/openImageNet/dataset/train/d279808768538a7b.jpg"
    km.doCompress(filepath)


def test_showYUV():
    filepath =  "/mnt/nas/openImageNet/dataset/train/1ebb35691b23f11e.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/4728d246e8333a50.jpg"
    filepath = "/mnt/nas/openImageNet/dataset/train/d279808768538a7b.jpg"
    km.showYUV(filepath)

# test_show_image()
# test_resizeImage()
test_doCompress()
# test_showYUV()