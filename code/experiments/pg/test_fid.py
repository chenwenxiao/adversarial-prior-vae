import sys,os
sys.path.append("..")
from utils import get_fid_google
import numpy as np
from PIL import Image
import tfsnippet as spt
import cv2

ori_dir = "/home/cwx17/data/test"

debug_dir = "./pic"

if __name__ == "__main__":

    (cifar_train,_),(_,_) = spt.datasets.load_cifar10()

    names = os.listdir(debug_dir)
    ori_images = []
    for name in names:
        big_image = Image.open(os.path.join(debug_dir,name))
        for x in range(10):
            for y in range(10):
                image = big_image.crop((x*32,y*32,(x+1)*32,(y+1)*32))
                ori_images.append(np.array(image))

    cifar_train = cifar_train[:len(ori_images)]
    ori_images = np.array(ori_images)

    print('shape of cifar',cifar_train.shape)
    print('shape of ori',ori_images.shape)
    # cv2.imshow("29",ori_images[29])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    get_fid_google(cifar_train,ori_images)