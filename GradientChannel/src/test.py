import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *

n_epochs = 10000
learning_rate_val = 0.001
weight_decay_rate = 0.00001
momentum = 0.9
batch_size = 400
lambda_recon = 0.999
lambda_adv = 0.001

overlap_size = 7
hiding_size = 64

testset_path  = '../data/lsun_testset.pickle'
result_path= '../results/lsun/'
pretrained_model_path = '../models/lsun/model-1'
# testset = pd.read_pickle( testset_path )

is_train = tf.placeholder( tf.bool )
images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")

# model = Model()

# reconstruction = model.build_reconstruction(images_tf, is_train)

sess = tf.InteractiveSession()

# t1 = [[1, 2, 3], [4, 5, 6], [7,8,9]]
t1 = [[[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]], [[7,8,9],[7,8,9],[7,8,9]]], 
    [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]], [[7,8,9],[7,8,9],[7,8,9]]]]

t2 = [[[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]], [[7,8,9],[7,8,9],[7,8,9]]], 
    [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]], [[7,8,9],[7,8,9],[7,8,9]]]]

t3 = tf.concat(3, [t1, t2])

shape_t1 = tf.shape(t3)

tf.initialize_all_variables().run()

shape_t1_out = sess.run(shape_t1)

# print(t3_out)
print(shape_t1_out)

# t4 = [[1, 1, 1], 
#     [2, 2, 2], 
#     [3,3,3]]

# tf.stack




