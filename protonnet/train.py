from parameters import *
from models.proton_net import proton_net
from metrics.triplet_loss import triplet_loss
from utils.datagenerator import data_generator
import tensorflow as tf
import os


print(f"""
	*---------------------------------------*
	Author: Muhammad Hanan Asghar
	Model: ProtonNet
	Type: Recognition Tasks
	Date: 12-03-2022
	Language: Python
	Tensorflow Version: {tf.__version__}
	Batch Size: {BATCH_SIZE}
	Image Size: {IMAGE_SIZE}
	Embedding Size: {EMB_SIZE}
  	Disclaimer: I have tested if you got any error related to datagenerator then make sure that you have minimum to two images per directory.
  Otherwise it will give error.
	*---------------------------------------*
	""")

# PROTON MODEL
protonnet = proton_net()

path = os.path.join(os.getcwd(), "weights")
if os.path.exists(path):
  protonnet = tf.keras.models.load_model(os.path.join(path, "ProtonNet"), custom_objects={'triplet_loss': triplet_loss})
  print("Weights Loaded.")
else:
  opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  protonnet.compile(loss=triplet_loss, optimizer=opt)
  print("New Model Compiled.")

checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH_H5, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS)
callbacks = [checkpoint, tensorboard]

history = protonnet.fit(
    data_generator(batch_size=BATCH_SIZE),
    epochs = EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCHS,
    verbose=1,
    callbacks = callbacks
)
