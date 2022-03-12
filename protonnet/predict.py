from parameters import *
import tensorflow as tf
from metrics.triplet_loss import triplet_loss



proton_net = tf.keras.models.load_model(MODEL_SAVE_PATH_H5,
                                        custom_objects={'triplet_loss': triplet_loss})

protonnet = tf.keras.models.Model(inputs=[proton_net.layers[3].input], outputs=[proton_net.layers[3].output])

def predict(positive_path, negative_path, model, triplet):
  def _image(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr,(128, 128)).numpy()
    image_arr = image_arr.astype("float32")
    image_arr = image_arr / 255.
    return image_arr
  anchor = _image(positive_path)
  anchor = tf.expand_dims(anchor, 0)
  positive = _image(positive_path)
  positive = tf.expand_dims(positive, 0)
  negative = _image(negative_path)
  negative = tf.expand_dims(negative, 0)
  y_pred = model.predict([anchor, positive, negative])
  loss = triplet(y_pred).numpy()
  if loss[0] == 0:
    return False
  return True


prediction = predict("/content/madhuri.jpg", "/content/sunny_leone.png", proton_net, triplet_loss)