import rate_control_game as rcg
import cv2
import tqdm
# import fiftyone
from openimages.download import download_images

image_dir = '../../content/dataset'
rateControlGame = rcg.RateControlGame(image_dir,None)

def showImage():
    imageFile = '/Users/lert-hg/Documents/PhD/reinforcement/DRL/content/TinyImageNet/train/0/0_7.jpg'
    # /Users/lert-hg/Documents/PhD/reinforcement/DRL/content/dataset_split/test/plane/plane.jpg
    frameImage = cv2.imread(imageFile)

    # Window name in which image is displayed
    window_name = 'image'
    
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow(window_name, frameImage)
    
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 

    pic_height, pic_width, _ = frameImage.shape
    print(frameImage.shape)


def fiftyoneDatasetDownloader():
    dataset = fiftyone.zoo.load_zoo_dataset(
                "open-images-v6",
                split="validation",
                label_types=["detections", "segmentations"],
                classes=["Cat", "Dog"],
                max_samples=100,
            )

    session = fiftyone.launch_app(dataset)


def downloadOpenImages(dest,classes):
    image_dir = '../../content/dataset'
    classes = ["Helmet","Taxi","Car","Backpack","Bicycle"]
    download_images(image_dir, classes, None)

