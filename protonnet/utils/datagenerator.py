import tensorflow as tf
import numpy as np
import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

from parameters import *
sys.path.remove(str(_parentdir))

def data_generator(batch_size = 2):
  def _image(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr,(IMAGE_SIZE[0], IMAGE_SIZE[1])).numpy()
    image_arr = image_arr.astype("float32")
    image_arr = image_arr / 255.
    return image_arr
  
  while True:
    anchors = []
    positives = []
    negatives = []
    for _ in range(batch_size):
      pos_idx = random.randint(0, len(DATASET_DIRS)-1)
      anchor_directory = os.path.join(DATASET, DATASET_DIRS[pos_idx])
      anchor_images = np.array(os.listdir(anchor_directory))
      anchor_image = np.random.choice(anchor_images)
      positive_image = anchor_images[np.random.choice(np.where(anchor_images != anchor_image)[0])]
      positive_image_nmp = _image(os.path.join(anchor_directory, positive_image))
      anchor_image_nmp = _image(os.path.join(anchor_directory, anchor_image))

      neg_idx = np.random.choice(np.where(DATASET_DIRS != anchor_directory)[0])
      negative_directory = os.path.join(DATASET, DATASET_DIRS[neg_idx])
      negative_images = np.array(os.listdir(negative_directory))
      negative_image = np.random.choice(negative_images)
      negative_image_nmp = _image(os.path.join(negative_directory, negative_image))
      # print(anchor_image, positive_image, negative_image)
      anchors.append(anchor_image_nmp)
      positives.append(positive_image_nmp)
      negatives.append(negative_image_nmp)
    yield ([np.array(anchors), np.array(positives), np.array(negatives)], np.zeros((batch_size, 1)).astype("float32"))