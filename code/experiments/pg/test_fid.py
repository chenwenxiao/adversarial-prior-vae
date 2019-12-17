import sys,os
sys.path.append("..")
from utils import get_fid_google,graph_def
import numpy as np
from PIL import Image
import tfsnippet as spt
import cv2
import tensorflow as tf

tfgan = tf.contrib.gan



ori_dir = "/home/cwx17/data/test"

debug_dir = "./pic"

def inception_transform(inputs):
  with tf.control_dependencies([
      tf.assert_greater_equal(inputs, 0.0),
      tf.assert_less_equal(inputs, 255.0)]):
    inputs = tf.identity(inputs)
  preprocessed_inputs = tf.map_fn(
      fn=tfgan.eval.preprocess_image, elems=inputs, back_prop=False)
  return tfgan.eval.run_inception(
      preprocessed_inputs,
      graph_def=graph_def,
      output_tensor=["pool_3:0", "logits:0"])

def inception_transform_np(inputs, batch_size):
  """Computes the inception features and logits for a given NumPy array.
  The inputs are first preprocessed to match the input shape required for
  Inception.
  Args:
    inputs: NumPy array of shape [-1, H, W, 3].
    batch_size: Batch size.
  Returns:
    A tuple of NumPy arrays with Inception features and logits for each input.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    inputs_placeholder = tf.placeholder(
        dtype=tf.float32, shape=[None] + list(inputs[0].shape))
    features_and_logits = inception_transform(inputs_placeholder)
    features = []
    logits = []
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))
    for i in range(num_batches):
      input_batch = inputs[i * batch_size:(i + 1) * batch_size]
      x = sess.run(
          features_and_logits, feed_dict={inputs_placeholder: input_batch})
      features.append(x[0])
      logits.append(x[1])
    features = np.vstack(features)
    logits = np.vstack(logits)
    return features, logits

def get_fid_test(sample, real):
    """Returns the FID based on activations.

    Args:
      fake_activations: NumPy array with fake activations.
      real_activations: NumPy array with real activations.
    Returns:
      A float, the Frechet Inception Distance.
    """
    tf.import_graph_def(graph_def, name='FID_Inception_Net')
    with tf.Session() as sess:
        (fake_activations,_) = inception_transform_np(sample, 50)
        (real_activations,_) = inception_transform_np(real, 50)
        fake_activations = tf.convert_to_tensor(fake_activations)
        real_activations = tf.convert_to_tensor(real_activations)
        fid = tfgan.eval.frechet_classifier_distance_from_activations(
            real_activations=real_activations,
            generated_activations=fake_activations)
        fid = sess.run(fid)
    return fid

if __name__ == "__main__":

    (cifar_train,_),(cifar_test,_) = spt.datasets.load_cifar10()

    names = os.listdir(ori_dir)
    ori_images = []
    for name in names:
        big_image = Image.open(os.path.join(ori_dir,name))
        for x in range(10):
            for y in range(10):
                image = big_image.crop((x*32,y*32,(x+1)*32,(y+1)*32))
                ori_images.append(np.array(image))

    cifar_train = cifar_train[:len(ori_images)]
    cifar_test = cifar_test[:len(ori_images)]
    ori_images = np.array(ori_images)

    print('shape of cifar',cifar_train.shape)
    print('shape of ori',ori_images.shape)
    # cv2.imshow("29",ori_images[29])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    fid1 = get_fid_test(ori_images,cifar_train)
    fid3 = get_fid_test(cifar_test,cifar_train)

    fid0 = get_fid_google(ori_images,cifar_train)
    fid2 = get_fid_google(cifar_test,cifar_train)

    print(f"FID google/test: {fid0}/{fid1}")
    print(f"FID google/test: {fid2}/{fid3}")