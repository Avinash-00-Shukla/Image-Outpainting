# -*- coding: utf-8 -*-
"""
@author: avina
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PIL import Image
import model
import util
import os
import sys

#%%
model_PATH = "output\models\model100.ckpt"
in_PATH = 'a.jpg'
out_PATH = 'CALC_IMG'

tf.reset_default_graph()

IMAGE_SZ = 128

img = np.array(Image.open(in_PATH).convert('RGB'))[np.newaxis] / 255.0
img_p = util.preprocess_images_outpainting(img)

G_Z = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 4], name='G_Z')
G_sample = model.generator(G_Z)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_PATH)
    output, = sess.run([G_sample], feed_dict={G_Z: img_p})
    util.save_image(output[0], out_PATH)
