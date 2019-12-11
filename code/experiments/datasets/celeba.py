import requests
import zipfile,os
import tfsnippet as spt
from tfsnippet import DataFlow
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random
import cv2 as cv


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

DEBUG_IMG = '/Users/lwd/Downloads/img_align_celeba'
DEBUG_EVAL = '/Users/lwd/Downloads/list_eval_partition.txt'

IMG_ZIP_PATH = '/home/cwx17/data/celeba/img_align_celeba.zip'
IMG_PATH = '/home/cwx17/data/celeba/img_align_celeba'
EVAL_PATH = '/home/cwx17/data/celeba/eval.txt'

def _fetch_array_x(path):
    file_names = os.listdir(path)
    file_names.sort()
    imgs = []
    if debug:
        cnt = 0
        for name in file_names:
            print(name)
            u_img = mpimg.imread(path+'/'+name)
            u_img = u_img[20:-20]
            u_img = cv.resize(u_img,dsize=(32,32),interpolation=cv.INTER_LINEAR)
            imgs.append(u_img)
            cnt+=1
            if cnt == 100:
                break
    else:
        for name in file_names:
            u_img = mpimg.imread(path+'/'+name)
            u_img = u_img[20:-20]
            u_img = cv.resize(u_img,dsize=(32,32),interpolation=cv.INTER_LINEAR)
            imgs.append(u_img)
        
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

debug = False


def load_celeba(x_shape=(218, 178), x_dtype=np.float32, y_dtype=np.int32,
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
    if debug:
        if not os.path.exists(DEBUG_EVAL):
            print(f'eval doesn\'t exist\ndownloading eval \ndst: {DEBUG_EVAL}')
            misc.download_celeba_eval(DEBUG_EVAL);
            print('downloaded')
        x = _fetch_array_x(DEBUG_IMG).astype(x_dtype)
        y = _fetch_array_y(DEBUG_EVAL).astype(y_dtype)
    else:   
        if not os.path.exists(IMG_PATH):
            print('img not exist')
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

        x = _fetch_array_x(IMG_PATH).astype(x_dtype)
        y = _fetch_array_y(EVAL_PATH).astype(y_dtype)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(x)):
        if random.random()<0.8 :
            train_x.append(x[i])      
            train_y.append(y[i])
        else:
            test_x.append(x[i])      
            test_y.append(y[i])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    if normalize_x:
        train_x /= np.asarray(255., dtype=x.dtype)
        test_x /= np.asarray(255., dtype=x.dtype)

    return (train_x, train_y), (test_x, test_y)

if __name__ == '__main__':
    (_x_train, _y_train), (_x_test, _y_test) = load_celeba()
    print(_x_train.shape)
    print(_x_test.shape)

    im = np.array(_x_train[19])
    im /= np.asarray(255., dtype=np.int32)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    print(im)
    plt.imshow(im)
    plt.show()