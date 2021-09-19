import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from scipy.io import loadmat
from scipy import misc
from PIL import Image
import os
from pathlib import Path
import torch
import random
import cv2

def read_image(filepath):
    
    # loading the png image as a 3d matrix
    # img = misc.imread(filepath)
    
    img = mpimg.imread(filepath)

    # uncomment the below code to view the loaded image
    # plt.imshow(img) # plotting the image
    # plt.show()
    
    # # scaling it so that the values are small
    img = img / 255

    return img

def resizeImage(infile, basewidth=128):

    global SAVE_DIR_NAME
    img = Image.open(infile)
    print(img)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    size = basewidth, hsize

    # outfile = os.path.splitext(infile)[0] + ".thumbnail"
    filename = Path(infile).stem
    outPng = f"{SAVE_DIR_NAME}/input-{filename}.png"
    outJpg = f"{SAVE_DIR_NAME}/input-{filename}.jpg"
    print(outPng)
    if infile != outPng:
        try:
            img.thumbnail(size, Image.ANTIALIAS)
            # im.save(outPng, "JPEG")
            # plt.imshow(img) # plotting the image
            # plt.show()
        except IOError:
            print (f"cannot create thumbnail filepath: {infile}")

    img = np.asarray(img)
    saveImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # saveImage = img.float()
    cv2.imwrite(outPng, saveImage)
    cv2.imwrite(outJpg, saveImage)

    img = img / 255
    return img


def preprocessData(image):
    # turn array into tensor
    img = torch.tensor(image).float().to(DEVICE)

    # preprocess data
    img = img.view(-1, 3)

    return img


def kMeansInitCentroids(dataset, K):
    # define upper bound
    upper_bound = dataset.shape[0]
    # generate random number in the range[0,upper_bound)
    index = random.sample(range(0, upper_bound), K)
    return index, dataset[index]

# Integrate each module
def k_means(dataset, K, num_epochs):
    _, centroids = kMeansInitCentroids(dataset, K)
    for epoch in range(num_epochs):
        print(f"RUN epoch - {epoch}")
        idx = findClosestCentroids(dataset, centroids)
        centroids = computeMeans(dataset, idx, K)
    return centroids, idx

# Assign each data to the corresponding centroids
def findClosestCentroids(dataset, centroids):
    """
    return:
        idx:the result of assignment.
        idx[i] is a list,containing the datas been assigned to ith centroids
    """
    # k:the number of centroids
    k = centroids.shape[0]
    # implement vectorization by broadcasting
    distance = (dataset.unsqueeze(1) - centroids.unsqueeze(0)) ** 2
    distance = distance.sum(dim=2)
    
    # assign each data
    assignments = distance.argmin(dim=1)
    # create k empty list
    idx = [[] for _ in range(k)]
    
    # store assignment result
    data_idx = 0
    for assignment in assignments:
        idx[assignment.item()].append(data_idx)
        data_idx += 1
    
    return idx

# compute means based on centroids
def computeMeans(dataset, idx, K):
    centroids = [[] for _ in range(K)]
    for k in range(K):
        centroids[k].append(dataset[idx[k]].mean(dim=0))
    centroids = torch.cat([centroids[i][0] for i in range(K)]).float().to(DEVICE)
    return centroids.view(K, -1)

def compress_image(centroids, idx, shape, clusters, filepath=""):

    global SAVE_DIR_NAME
     # create the compressed image
    recovered = torch.ones(shape[0]*shape[1], shape[2]).float().to(DEVICE)
    print(f"recovered.dtype - {recovered.dtype}")
    print(f"centroids.dtype - {centroids.dtype}")

    for k in range(clusters):
        # set the color of each pixel accroding its centroids
        recovered[idx[k]] = centroids[k]
    recovered = recovered.view(shape)
    recovered = recovered.cpu().data.numpy()

    # show the image
    # plt.imshow(recovered)
    # plt.show()
    # recovered *= 255

    filename = Path(filepath).stem
    savePng = f"{SAVE_DIR_NAME}/compressed_{str(clusters)}-{filename}.png"
    saveJpg = f"{SAVE_DIR_NAME}/compressed_{str(clusters)}-{filename}.jpg"
    # saving the compressed image.
    # misc.imsave('compressed_' + str(clusters) +
    #                     '_colors.png', recovered)
    print(savePng)
    img = cv2.convertScaleAbs(recovered, alpha=(255.0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(savePng, img)
    cv2.imwrite(saveJpg, img)


def showYUV(filepath):
    image = cv2.imread(filepath)
    cv2.imshow("original",image)
    cv2.imwrite("original.png", image)
    cv2.imwrite("original.jpg", image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    image[:,:,0] = image[:,:,0]-30
    # image[1::] = image[0::]-10
    print(image)

    cv2.imshow("yuv",image)
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0) 
    
    # #closing all open windows 
    # cv2.destroyAllWindows() 

    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    cv2.imshow("bgr",image)
    cv2.imwrite("yuv.png", image)
    cv2.imwrite("yuv.jpg", image)
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 

BASE_WIDTH = 256
CLUSTERS = 8
ITERATIONS = 5
DEVICE = "cuda:0"
DEVICE = "cpu"

SAVE_DIR_NAME=""

def doCompress(filepath):

    global SAVE_DIR_NAME
    SAVE_DIR_NAME = Path(filepath).stem
    print(f"doCompress - {filepath}")
    try:
        # Create target Directory
        os.mkdir(SAVE_DIR_NAME)
        print("Directory " , SAVE_DIR_NAME ,  " Created ") 
    except FileExistsError:
        print("Directory " , SAVE_DIR_NAME ,  " already exists")

    # img = read_image(filepath)
    img = resizeImage(filepath,BASE_WIDTH)
    print(f"resize - {img.shape}")
    output_shape = img.shape

    img = preprocessData(img)
    print(img.shape)
    print(f"preprocessData - {img.shape}")

    cluster = CLUSTERS
    for i in range(0,5):
        means, index = k_means(img, cluster, ITERATIONS)
        compress_image(means, index, output_shape, cluster, filepath)
        cluster *= 2