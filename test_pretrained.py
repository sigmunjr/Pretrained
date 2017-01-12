import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from tensorflow.contrib.slim import nets
import tensorflow.contrib.slim as slim
from skimage.io import imread
import os
from IndexToLabel import label

def load_and_preprocess(image_name):
  #subtrackting ImageNet mean (only for ResNet)
  return imresize(imread(image_name), [299, 299]).astype(np.float32) - np.array([123.68, 116.779, 103.939])

#input image
image = tf.placeholder(tf.float32, shape=[299, 299, 3])

#build network
with slim.arg_scope(nets.resnet_v1.resnet_arg_scope(is_training=False)):
  raw_output, end_points = nets.resnet_v1.resnet_v1_50(image[tf.newaxis, :, :, :], num_classes=1000)
predictions = end_points['predictions']
after_first_resnet_unit = end_points['resnet_v1_50/block1/unit_1/bottleneck_v1']
spatial_first_weight = tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights:0')

#Load pretrained weights
saver = tf.train.Saver(var_list=tf.global_variables())
sess = tf.Session()
saver.restore(sess, os.path.abspath('resnet_v1_50.ckpt'))

#test network
for img_name in os.listdir('./images'):
  tmp_img = load_and_preprocess('images/' + img_name)
  label_index, filter_img, weight_img = sess.run([tf.squeeze(tf.argmax(predictions, axis=3)),
                                                  after_first_resnet_unit,
                                                  spatial_first_weight], feed_dict={image: tmp_img})
  label_name = label(label_index)
  for i in range(8):
    for j in range(8):
      plt.figure(1)
      plt.subplot(8, 8, j + i*8 + 1)
      plt.imshow(weight_img[:, :, 32, j + i*j], vmin=filter_img[0, :, :, j + i*j].min(), vmax=filter_img[0, :, :, j + i*j].max())
      plt.figure(2)
      plt.subplot(8, 8, j + i*8 + 1)
      plt.imshow(filter_img[0, :, :, j + i*j])
  plt.show()
  print(img_name, 'classified as:', label_name)
