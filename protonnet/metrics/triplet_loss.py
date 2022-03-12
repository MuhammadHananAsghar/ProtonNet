import tensorflow as tf
import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

from parameters import *
sys.path.remove(str(_parentdir))



EMB_SIZE1 = EMB_SIZE*2
EMB_SIZE2 = EMB_SIZE1*2

def triplet_loss(y_true, y_pred):
  anchor = y_pred[:,:EMB_SIZE]
  positive = y_pred[:,EMB_SIZE:EMB_SIZE1]
  negative = y_pred[:,EMB_SIZE1:EMB_SIZE2]

  positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
  negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
  return tf.maximum(positive_dist - negative_dist + ALPHA, 0.)