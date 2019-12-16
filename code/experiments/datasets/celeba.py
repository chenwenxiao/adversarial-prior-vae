'''
load CelebA dataset as numpy array

usage:

    import celeba

    (train_x, train_y), (test_x, test_y) = load_celeba()

'''
from PIL import Image
from scipy.ndimage import filters
from scipy.misc import imresize, imsave
import requests
import zipfile,os
import tfsnippet as spt
from tfsnippet import DataFlow
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random
import cv2 as cv
import shutil

from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from skimage import transform, filters
from tqdm import tqdm

from .base import StandardImageDataSet


def download_file_from_google_drive(id, destination):
    # usage : download_file_from_google_drive(file_id_on_google_drive, path)
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  

class misc():
    @staticmethod 
    def download_celeba_img(path):
        'url of aligned & cropped celeba https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
        ' size: 218*178'
        ' format: jpg'
        download_file_from_google_drive('0B7EVK8r0v71pZjFTYXZWM3FlRnM',path)

    @staticmethod 
    def download_celeba_eval(path):
        'url of celeba eval https://drive.google.com/open?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk'
        download_file_from_google_drive('0B7EVK8r0v71pY0NSMzRuSXJEVkk',path)

    @staticmethod
    def unzip(src,dest):
        '''
        src: address of the zip
        dest: a directory to store the file
        '''
        f = zipfile.ZipFile(src)
        if not os.path.exists(dest):
            os.makedirs(dest)
        f.extractall(dest)  

    celba_size = 202598;
    @staticmethod
    def get_tt():
        train=[]
        test=[]
        for i in range(202598):
            random.seed(i)
            if (random.random()>0.83333):
                test.append(i)
            else:
                train.append(i)
        return train,test

DEBUG_IMG = '/Users/lwd/Downloads/img_align_celeba'
DEBUG_EVAL = '/Users/lwd/Downloads/list_eval_partition.txt'

IMG_ZIP_PATH = '/home/cwx17/data/celeba/img_align_celeba.zip'
IMG_PATH = '/home/cwx17/data/celeba/img_align_celeba'
EVAL_PATH = '/home/cwx17/data/celeba/list_eval_partition.txt'
MAP_DIR_PATH = '/home/cwx/data'
MAP_PATH = '/home/cwx/data/CelebA'

# TRAIN_DIR_PATH = '/home/cwx17/data/celeba/train'
# TRAIN_X_PATH = '/home/cwx17/data/celeba/train/img'
# TRAIN_Y_PATH = '/home/cwx17/data/celeba/train/train_eval.txt'
# TRAIN_INFO_PATH = '/home/cwx17/data/celeba/train/info.txt'
# TRAIN_X_ARR_PATH = '/home/cwx17/data/celeba/train/imgarr.npy'


# TEST_DIR_PATH = '/home/cwx17/data/celeba/test'
# TEST_X_PATH = '/home/cwx17/data/celeba/test/img'
# TEST_Y_PATH = '/home/cwx17/data/celeba/test/test_eval.txt'
# TEST_INFO_PATH = '/home/cwx17/data/celeba/test/info.txt'
# TEST_X_ARR_PATH = '/home/cwx17/data/celeba/test/imgarr.npy'

def _fetch_array_x(path):
    file_names = os.listdir(path)
    file_names.sort()
    imgs = []
    scale = 148 / float(64)
    sigma = np.sqrt(scale) / 2.0
    for name in file_names:
        im = Image.open(os.path.join(path,name))
        im = im.crop((15,40,163,188))
        img = np.asarray(im)
        img.setflags(write=True)
        for dim in range(img.shape[2]):
            img[...,dim] = filters.gaussian_filter(img[...,dim], sigma=(sigma,sigma))
        img = imresize(img,(64,64,3))
        imgs.append(img)
        
    return np.array(imgs)

def _fetch_array_y(path):
    evalue = []
    with open(path,'rb') as f:
        if debug:
            cnt =0
            for line in f.readlines():
                print(line.decode('utf-8'))
                q = line.decode('utf-8')
                q = q.strip()
                print(q.split(' '))
                q = int(q.split(' ')[1])
                evalue.append(q)
                cnt +=1 
                if cnt == 100:
                    break
        else:
            for line in f.readlines():
                q = line.decode('utf-8')
                q = q.strip()
                q = int(q.split(' ')[1])
                evalue.append(q)
    return np.array(evalue)

def _store_array_x(arr,path,prefix):
    cnt=1;
    if not os.path.exists(path):
        os.makedirs(path)
    for img in arr:
        mpimg.imsave(f'{path}/{prefix}{cnt}.jpg',img)
        cnt+=1

def _store_array_y(arr,path,dir_path,prefix):
    cnt = 1;
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path,'w+') as f:
        for i in arr:
            f.write(f"{prefix}{cnt} {i}\r\n")
            cnt+=1
        
debug = False


__all__ = ['CelebADataSet']


def load_celeba(mmap_base_dir = MAP_PATH, img_size = 64):
    if mmap_base_dir is None:
        raise ValueError('`mmap_base_dir` is required for CelebA.')
    if img_size not in (32, 64):
        raise ValueError(f'`img_size` must be either 32 or 64: got {img_size}.')
    data_dir = os.path.join(mmap_base_dir, 'CelebA')
    pfx = f'{img_size}x{img_size}'
    train_x = np.memmap(
        os.path.join(data_dir, f'{pfx}/train.dat'), dtype=np.uint8,
        mode='r', shape=(162770, img_size, img_size, 3))
    valid_x = np.memmap(
        os.path.join(data_dir, f'{pfx}/val.dat'), dtype=np.uint8,
        mode='r', shape=(19867, img_size, img_size, 3))
    test_x = np.memmap(
        os.path.join(data_dir, f'{pfx}/test.dat'), dtype=np.uint8,
        mode='r', shape=(19962, img_size, img_size, 3))
    return train_x, valid_x, test_x


def _resize(img, img_size=64, bbox=(40, 218-30, 15, 178-15)):
    # this function is copied from:
    # https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py

    img = img[bbox[0]: bbox[1], bbox[2]: bbox[3]]

    # Smooth image before resize to avoid moire patterns
    scale = img.shape[0] / float(img_size)
    sigma = np.sqrt(scale) / 2.0
    img = img.astype(np.float32) / 255.
    img = filters.gaussian(img, sigma=sigma, multichannel=True)
    img = transform.resize(
        img, (img_size, img_size, 3), order=3,
        # Turn off anti-aliasing, since we have done gaussian filtering.
        # Note `anti_aliasing` defaults to `True` until skimage >= 0.15,
        # which version is released in 2019/04, while the repo
        # `andersbll/autoencoding_beyond_pixels` was released in 2015.
        anti_aliasing=False,
        # same reason as above
        mode='constant',
    )
    img = (img * 255).astype(np.uint8)
    return img


class CelebADataSet(StandardImageDataSet):

    def __init__(self,
                 image_size: int = 64,
                 mmap_base_dir: Optional[str] = None,
                 use_validation: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        super(CelebADataSet, self).__init__(
            name='CelebA',
            mmap_base_dir=mmap_base_dir,
            loader_fn=partial(load_celeba, img_size=image_size),
            color_depth=256,
            has_y=False,
            valid_data_count=19867 if use_validation else 0,
            random_state=random_state,
        )

        assert (self.value_shape == (image_size, image_size, 3))
        if use_validation:
            assert (self.train_data_count == 162770)
            assert (self.valid_data_count == 19867)
        else:
            assert (self.train_data_count == 182637)
            assert (self.valid_data_count == 0)
        assert (self.test_data_count == 19962)

    @staticmethod
    def make_mmap(source_dir: str,
                  mmap_base_dir: str,
                  force: bool = False):
        """
        Generate the mmap files.

        The image pre-processing method is the same as
        `https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py`.

        Args:
            source_dir: The root directory of the original CelebA dataset.
                The following directory and file are expected to exist:
                * aligned images: `source_dir + "/Img/img_align_celeba"`
                * partition file: `source_dir + "/Eval/list_eval/partition.txt"`
            mmap_base_dir: The mmap base directory.
            force: Whether or not to force generate the files even if they
                have been already generated?
        """
        # check file system paths
        image_dir = os.path.join(source_dir, 'img_align_celeba')
        partition_file = os.path.join(
            source_dir, 'list_eval_partition.txt')

        target_dir = os.path.join(mmap_base_dir, 'CelebA')

        # read the partition file
        df = pd.read_csv(partition_file,
                         sep=' ', header=None,
                         names=['file_name', 'set_id'],
                         dtype={'file_name': str, 'set_id': int},
                         engine='c')
        assert(len(df[df['set_id'] == 0]) == 162770)
        assert(len(df[df['set_id'] == 1]) == 19867)
        assert(len(df[df['set_id'] == 2]) == 19962)

        # process the images
        def process_set(set_id, target_file, img_size):
            df_set = df[df['set_id'] == set_id]
            set_length = len(df_set)
            image_shape = (img_size, img_size, 3)
            parent_dir = os.path.split(os.path.join(target_dir, target_file))[0]
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            processed_file = os.path.join(
                target_dir, target_file + '.processed')
            if not force and os.path.isfile(processed_file):
                return
            mmap_arr = np.memmap(
                os.path.join(target_dir, target_file),
                dtype=np.uint8,
                mode='w+',
                shape=(set_length,) + image_shape,
            )

            for i, (_, row) in enumerate(
                    tqdm(df_set.iterrows(), total=set_length,
                         ascii=True, desc=target_file, unit='image')):
                file_path = os.path.join(image_dir, row['file_name'])

                # read image into array, according to the method of:
                # https://github.com/andersbll/deeppy/blob/master/deeppy/dataset/celeba.py
                im = PILImage.open(file_path)
                im_arr = im_bytes = None
                try:
                    width, height = im.size
                    im_bytes = im.tobytes()
                    im_arr = np.frombuffer(im_bytes, dtype=np.uint8). \
                        reshape((height, width, 3))
                    mmap_arr[i, ...] = _resize(im_arr, img_size=img_size)
                finally:
                    im.close()
                    del im_arr
                    del im_bytes
                    del im

            # if all is okay, generate the processed file
            with open(processed_file, 'wb') as f:
                f.write(b'\n')

        for s in (32, 64):
            pfx = f'{s}x{s}'
            process_set(2, f'{pfx}/test.dat', s)
            process_set(1, f'{pfx}/val.dat', s)
            process_set(0, f'{pfx}/train.dat', s)


def prepare_celeba(x_shape=(218, 178), x_dtype=np.float32, y_dtype=np.int32,
               normalize_x=False):
    """
    Load the CelebA dataset as NumPy arrays.
    samilar to load_not_mnist

    Args:
        Unimplemented!(haven't found a good way to resize) x_shape: Reshape each digit into this shape.  Default ``(218, 178)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
             
    """
    if (not os.path.exists(IMG_PATH)) or (len(os.listdir(IMG_PATH))!=202598):
        print('img file not exist or img num wrong')
        if os.path.exists(IMG_ZIP_PATH):
            print(f'zipped file exists\n unzipping\ndst: {IMG_PATH}')
            misc.unzip(IMG_ZIP_PATH,IMG_PATH)
            print('unzipped')
        else:
            print(f'zipped file dosen\'t exist\ndownloading img \ndst: {IMG_ZIP_PATH}')
            misc.download_celeba_img(IMG_ZIP_PATH);
            print(f'downloaded\nstart unzip\ndst: {IMG_PATH}')
            misc.unzip(IMG_ZIP_PATH,IMG_PATH)
            print('unzipped')
    if not os.path.exists(EVAL_PATH):
        print(f'eval doesn\'t exist\ndownloading eval \ndst: {EVAL_PATH}')
        misc.download_celeba_eval(EVAL_PATH);
        print('downloaded')

        # x = _fetch_array_x(IMG_PATH).astype(x_dtype)
        # y = _fetch_array_y(EVAL_PATH).astype(y_dtype)

        # train_x = []
        # train_y = []
        # test_x = []
        # test_y = []

        # for i in train:
        #     train_x.append(x[i])
        #     train_y.append(y[i])

        # for i in test:
        #     test_x.append(x[i])
        #     test_y.append(y[i])

        # train_x = np.array(train_x)
        # train_y = np.array(train_y)
        # test_x = np.array(test_x)
        # test_y = np.array(test_y)

        # train_x /= np.asarray(255., dtype=x.dtype)
        # test_x /= np.asarray(255., dtype=x.dtype)

        # np.save(train_x,TRAIN_X_ARR_PATH)
        # np.save(test_x,TEST_X_ARR_PATH)

    # train_x = np.load(TRAIN_X_ARR_PATH)
    # test_x = np.load(TEST_X_ARR_PATH)

    # return (train_x, train_y), (test_x, test_y)

if __name__ == '__main__':
    prepare_celeba()
    # CelebADataSet.make_mmap(IMG_PATH,MAP_DIR_PATH,True)
    # x_train,_,x_test=load_celeba()

    # print(x_train.shape)
    # print(x_test.shape)

    # im = np.array(x_train[19])
    # im /= np.asarray(255., dtype=np.int32)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    # print(im)
    # plt.imshow(im)
    # plt.show()