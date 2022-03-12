import tensorflow as tf
import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

from parameters import *
sys.path.remove(str(_parentdir))


def embedding_model():
	"""
	EMBEDDING MODEL
	"""
	input = tf.keras.layers.Input(shape=IMAGE_SIZE, name="input_image")
	l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(input)
	l2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(l1)

	l3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(l2)
	l4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(l3) 

	l5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(l4)
	l6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(l5) 

	l7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")(l6)
	l8 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(l7)

	l9 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")(l8)
	l10 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(l9) 

	flatten = tf.keras.layers.Flatten()(l10)
	dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
	dense1 = tf.keras.layers.BatchNormalization()(dense1)
	dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
	dense2 = tf.keras.layers.BatchNormalization()(dense2)
	output = tf.keras.layers.Dense(EMB_SIZE)(dense2)

	return tf.keras.models.Model(inputs=[input], outputs=[output], name=EMBEDDING_MODEL_NAME)