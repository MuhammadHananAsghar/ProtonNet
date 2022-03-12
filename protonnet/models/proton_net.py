import tensorflow as tf
import pathlib
from models.embedding import embedding_model
import sys

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

from parameters import *
sys.path.remove(str(_parentdir))


def proton_net():
	"""
	PROTON_NET
	"""
	embedding = embedding_model()
	input_anchor = tf.keras.layers.Input(shape=IMAGE_SIZE, name="anchor_input")
	input_positive = tf.keras.layers.Input(shape=IMAGE_SIZE, name="positive_input")
	input_negative = tf.keras.layers.Input(shape=IMAGE_SIZE, name="negative_input")


	embedding_anchor = embedding(input_anchor)
	embedding_positive = embedding(input_positive)
	embedding_negative = embedding(input_negative)

	output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)


	protonnet = tf.keras.models.Model(inputs=[input_anchor, input_positive, input_negative],
	                                    outputs=[output])
	return protonnet