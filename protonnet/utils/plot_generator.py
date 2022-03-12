import matplotlib.pyplot as plt
import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

from parameters import *
sys.path.remove(str(_parentdir))

def plot(generator):
  plt.figure(figsize=(10, 8))
  batch = next(generator)
  plt.subplot(1, 3, 1)
  plt.title("Anchor Image")
  plt.imshow(batch[0][0].reshape(IMAGE_SIZE))
  plt.subplot(1, 3, 2)
  plt.title("Positive Image")
  plt.imshow(batch[0][1].reshape(IMAGE_SIZE))
  plt.subplot(1, 3, 3)
  plt.title("Negative Image")
  plt.imshow(batch[0][2].reshape(IMAGE_SIZE))
  plt.show()