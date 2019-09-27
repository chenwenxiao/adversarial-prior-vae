import zipfile,os
#
import tensorflow as tf
from tensorflow.python.ops.gen_io_ops import read_file as tf_read_file
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.ops.iterator_ops import Iterator

#
import tfsnippet as spt
from tfsnippet import DataFlow
#
from pg import download_file_from_google_drive

assert tf.__version__ == '1.14.0'

class misc():

    @staticmethod 
    def download_celeba():
        'url of cropped celeba https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
        download_file_from_google_drive('0B7EVK8r0v71pZjFTYXZWM3FlRnM','./cropped_celeba.zip')

    @staticmethod
    def unzip(src,dest):
        '''
        src: address of the zip
        dest: a directory to store the file
        '''
        f = zipfile.ZipFile(src)
        if not os.path.exists(dest):
            os.mkdir(dest)
        f.extractall(dest)

class AlteredDataFlow(DataFlow):
    '''
    
    '''
    @staticmethod
    def input_fn(self,file_names):
        '''
        a generator for tf.data.Dataset.
        called when Dataset.prefetch's called 
        '''
        for single in file_names:
            single_img = tf_read_file(single)
            yield single_img
    
    def __iter__(self): 
        self.it = self.dataset.make_one_shot_iterator()
        yield self.it.get_next()

    def __init__(self,file_names,batch_size):
        '''
        Haven't seen the necessity to initialize spt.Dataflow
        Initialize for tf.data.Dataset
        '''
        self.dataset = Dataset.from_generator(AlteredDataFlow.input_fn,(tf.uint8))
        self.dataset = self.dataset.batch(batch_size)

    #override
    def map(self,fn):
        self.dataset = self.dataset.map(fn)

    def threaded(self,prefetch_num):
        self.dataset = self.dataset.prefetch(prefetch_num)
        return self

    

    
