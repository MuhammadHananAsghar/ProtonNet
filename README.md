# ProtonNet
A Network for Recognition Tasks

## Intuition
I got idea from FaceNet model and then trying to make and train Facenet like Model on my own. This model gives very good results even on training on Pins Face Dataset. This dataset contains only ~14K images of 105 different subjects.

### Heart
The heart of **ProtonNet** is Triplet Loss function which is proposed in FaceNet Paper by Google.
![image](https://user-images.githubusercontent.com/44013285/158017925-91d7400d-328d-431e-a243-2571f68ac2f1.png)

```python
def triplet_loss(y_true, y_pred):
  anchor = y_pred[:,:EMB_SIZE]
  positive = y_pred[:,EMB_SIZE:EMB_SIZE1]
  negative = y_pred[:,EMB_SIZE1:EMB_SIZE2]

  positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
  negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
  return tf.maximum(positive_dist - negative_dist + ALPHA, 0.)
```

### Usage
#### How to use?
Simple steps just change the Dataset_Path and Model_Save path in the ```params.py```.
```python
DATASET = ""
MODEL_SAVE_PATH_H5 = ""
```
### Pretrained Weights
#### Where to download pretrained weights?
Just go and download from these links
Weights(Training): 
```link
https://drive.google.com/file/d/1BC33NqYdJ39QKXiD5osHKY-0a37miWEx/view?usp=sharing
```
H5 Weights(Not For Training): 
```link
https://drive.google.com/file/d/1LNpZK0Kk3IiUB23kzWirYZhlDS-JQcbo/view?usp=sharing
```
Make weights folder in the protonnet folder and Copy weights folder ```ProtonNet``` in it.
```
Created with (Love). By Muhammad Hanan Asghar
```
