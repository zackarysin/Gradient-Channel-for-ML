import tensorflow as tf
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
import math

import cPickle

from util import *

def compute_single_channel_image_gradient(image_to_evaluate, is_stay_same_shape = False):

    if is_stay_same_shape:
        image_to_evaluate = tf.pad(image_to_evaluate, [[0,0],[1,1],[1,1],[0,0]], mode="SYMMETRIC")

    #image_to_evaluate= tf.Print(image_to_evaluate, ["shape=", tf.shape(image_to_evaluate)])

    with tf.variable_scope( "GradientColor" ):
        sobel_gx = tf.constant(
                value=[[0.5,1,0.5],[0,0,0],[-0.5,-1,-0.5]],
                dtype=tf.float32,
                shape=[3,3,1,1],
                name="Hor")

        sobel_gy = tf.constant(
                value=[[0.5,0,-0.5],[1,0,-1],[0.5,0,-0.5]],
                dtype=tf.float32,
                shape=[3,3,1,1],
                name="Ver")

    sobel_gx_conv = tf.nn.conv2d( image_to_evaluate, sobel_gx, [1,1,1,1], padding="VALID")
    sobel_gy_conv = tf.nn.conv2d( image_to_evaluate, sobel_gy, [1,1,1,1], padding="VALID")

    sobel_g = tf.sqrt(tf.add(tf.square(sobel_gx_conv), tf.square(sobel_gy_conv)))

    gradient = tf.mul(sobel_g,tf.sqrt(2.) / 8.0) # go from 0 to 1

    return gradient, sobel_g

def compute_image_gradient(image_to_evaluate, is_stay_same_shape = False):

    # gradient

    grayscale_images_tf = tf.image.rgb_to_grayscale(image_to_evaluate)

    # grayscale_img_sum = tf.summary.image("Grayscale", grayscale_images_tf)

    gradient, sobel_g = compute_single_channel_image_gradient(grayscale_images_tf, is_stay_same_shape)

    return gradient, sobel_g



