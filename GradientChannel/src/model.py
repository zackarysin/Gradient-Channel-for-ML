import tensorflow as tf
import numpy as np
import cv2
import math
import os
from glob import glob

import pandas as pd
import cPickle

from gradient import *
from util import *

class Model():

    def __init__(self, sess, config):

        self.sess = sess

        self.config = config

        if not os.path.exists(self.config.path_model):
            os.makedirs( self.config.path_model )

        if not os.path.exists(self.config.path_result):
            os.makedirs( self.config.path_result )

        if not os.path.exists( self.config.path_trainset_pickle ) or not os.path.exists( self.config.path_testset_pickle ):
            if self.config.path_trainset == self.config.path_testset:
                dataset_images = []
                for dir, _, _, in os.walk(self.config.path_trainset):
                    dataset_images.extend( glob( os.path.join(dir, '*')))

                dataset_images = np.hstack(dataset_images)

                self.trainset = pd.DataFrame({'image_path':dataset_images[:int(len(dataset_images)*self.config.config_trainset_testset_size_ratio)]})
                self.testset = pd.DataFrame({'image_path':dataset_images[int(len(dataset_images)*self.config.config_trainset_testset_size_ratio):]})

                self.trainset.to_pickle( self.config.path_trainset_pickle )
                self.testset.to_pickle( self.config.path_testset_pickle )
            else:
                trainset_images = []
                for dir, _, _, in os.walk(self.config.path_trainset):
                    trainset_images.extend( glob( os.path.join(dir, '*')))

                trainset_images = np.hstack(trainset_images)

                testset_images = []
                for dir, _, _, in os.walk(self.config.path_testset):
                    testset_images.extend( glob( os.path.join(dir, '*')))

                testset_images = np.hstack(testset_images)

                self.trainset = pd.DataFrame({'image_path':trainset_images[:]})
                self.testset = pd.DataFrame({'image_path':testset_images[:]})

                self.trainset.to_pickle( self.config.path_trainset_pickle )
                self.testset.to_pickle( self.config.path_testset_pickle )
        else:
            self.trainset = pd.read_pickle( self.config.path_trainset_pickle )
            self.testset = pd.read_pickle( self.config.path_testset_pickle )

        # using the last int(self.config.trainset_testset_size_ratio) % as testset
        self.testset.index = range(len(self.testset))
        self.testset = self.testset.ix[np.random.permutation(len(self.testset))]

        # variables for conveneince only. Should use self.config.image_size for instance

        self.build()

        pass


    def gradientLossBuild(self, imageToEvaluate):

        # images_tf_reshape = tf.reshape(self.images_tf, [3, self.batch_size, self.image_size, self.image_size])

        # grayscale_images_tf = tf.add(images_tf_reshape[0], images_tf_reshape[1])
        # grayscale_images_tf = tf.add(grayscale_images_tf, images_tf_reshape[2])
        # grayscale_images_tf = tf.mul(grayscale_images_tf, 1./3.)

        # grayscale_images_tf = tf.image.rgb_to_grayscale(self.images_tf)

        gradientPunitive, sobel_g = compute_gradient_image(imageToEvaluate)

        self.g_summaryList.append(tf.summary.histogram("gradient", gradientPunitive))

        gradientPunitive = tf.mul(gradientPunitive,math.pi/2.)
        gradientPunitive = tf.square(tf.sin(gradientPunitive))

        #sobel_gy_conv = tf.Print(sobel_gy_conv, [tf.shape(sobel_gy_conv)])

        # ori_img_sum = tf.summary.image("Ori", reconstruction)
        
        # hor_img_sum = tf.summary.image("Hor", sobel_gx)
        # ver_img_sum = tf.summary.image("Ver", sobel_gy)

        # sobel_gx_conv_img_sum = tf.summary.image("sobel_gx_conv", sobel_gx_conv)
        # sobel_gy_conv_img_sum = tf.summary.image("sobel_gy_conv", sobel_gy_conv)

        # gradient = tf.mul(tf.add(sobel_g, 1),1./2.)

        self.gradientLoss = tf.reduce_sum(gradientPunitive) / (self.config.size_batch * self.config.size_hiding * self.config.size_hiding )

        self.g_summaryList.append(tf.summary.image("sobel_g", sobel_g))
        self.g_summaryList.append(tf.summary.image("gradientPunitive", gradientPunitive))

        
        #self.g_summaryList.append(tf.summary.scalar("gradientLoss", self.gradientLoss))



    def build(self):

        self.input_image_depth = 4

        lambda_recon = self.config.lambda_recon
        lambda_adv = self.config.lambda_adv

        self.g_summaryList = []
        self.d_summaryList = []

        # each layer will do 128 x 128 -> 64 x 64
        encoderLayerNum = int(math.log(self.config.size_image) / math.log(2))
        encoderLayerNum = encoderLayerNum - 1 # minus 1 because the second last layer directly go from 4x4 to 1x1 
        print("encoderLayerNum=", encoderLayerNum)
        self.encoderLayerNum = encoderLayerNum

        decoderLayerNum = int(math.log(self.config.size_hiding) / math.log(2))
        decoderLayerNum = decoderLayerNum - 1
        print("decoderLayerNum=", decoderLayerNum)
        self.decoderLayerNum = decoderLayerNum

        self.writer = tf.summary.FileWriter("./../../ProjsStorage/ImageCompletion/logs", self.sess.graph)

        self.is_train = tf.placeholder( tf.bool )

        self.learning_rate = tf.placeholder( tf.float32, [])
        self.gradient_loss_mul = tf.placeholder( tf.float32, [])

        input_image_depth = self.input_image_depth

        self.images_tf = tf.placeholder( tf.float32, [self.config.size_batch, self.config.size_image, self.config.size_image, input_image_depth], name="images")

        # images_tf_rgb, images_tf_a = tf.split_v(3, [3, 1], self.images_tf)
        images_tf_rgb, images_tf_a = tf.split_v(self.images_tf, [3, 1], 3)

        gradient_images_tf_a, _ = compute_single_channel_image_gradient(images_tf_a, is_stay_same_shape=True)
        gradient_images_tf, _ = compute_image_gradient(images_tf_rgb, is_stay_same_shape=True)

        gradient_images_tf = tf.mul(tf.add(gradient_images_tf, gradient_images_tf_a), 0.5)

        images_tf_with_gradient_channel = tf.concat(3, [self.images_tf, gradient_images_tf])

        # self.images_tf = tf.Print(self.images_tf, ["Shape=", tf.shape(tf.concat(3, [self.images_tf, gradient_images_tf]))])

        labels_D = tf.concat( 0, [tf.ones([self.config.size_batch]), tf.zeros([self.config.size_batch])] )
        labels_G = tf.ones([self.config.size_batch])
        self.images_hiding = tf.placeholder( tf.float32, [self.config.size_batch, self.config.size_hiding, self.config.size_hiding, input_image_depth], name='images_hiding')

        reconstruction_ori, reconstruction = self.build_reconstruction(images_tf_with_gradient_channel, input_image_depth+1, ouput_image_depth=input_image_depth, is_train=self.is_train)

        image_hiding_rgb, image_hiding_a = tf.split_v(self.images_hiding, [3, 1], 3)
        reconstruction_rgb, reconstruction_a = tf.split_v(reconstruction, [3, 1], 3)

        gradient_image_hiding_a, _ = compute_single_channel_image_gradient(image_hiding_a, is_stay_same_shape=True)
        gradient_image_hiding_rgb, _ = compute_image_gradient(image_hiding_rgb, is_stay_same_shape=True)
        gradient_image_hiding = tf.mul(tf.add(gradient_image_hiding_rgb, gradient_image_hiding_a), 0.5)

        gradient_reconstruction_a, _ = compute_single_channel_image_gradient(reconstruction_a, is_stay_same_shape=True)
        gradient_reconstruction_rgb, _ = compute_image_gradient(reconstruction_rgb, is_stay_same_shape=True)
        gradient_reconstruction = tf.mul(tf.add(gradient_reconstruction_rgb, gradient_reconstruction_a), 0.5)

        images_hiding_with_gradient_channel = tf.concat(3, [self.images_hiding, gradient_image_hiding])
        reconstruction_with_gradient_channel = tf.concat(3, [reconstruction, gradient_reconstruction])

        adversarial_pos = self.build_adversarial(images_hiding_with_gradient_channel, input_image_depth+1, self.is_train)
        adversarial_neg = self.build_adversarial(reconstruction_with_gradient_channel, input_image_depth+1, self.is_train, reuse=True)
        adversarial_all = tf.concat(0, [adversarial_pos, adversarial_neg])

        # Applying bigger loss for overlapping region
        mask_recon = tf.pad(tf.ones([self.config.size_hiding - 2*self.config.size_overlap, self.config.size_hiding - 2*self.config.size_overlap]), [[self.config.size_overlap,self.config.size_overlap], [self.config.size_overlap,self.config.size_overlap]])
        mask_recon = tf.reshape(mask_recon, [self.config.size_hiding, self.config.size_hiding, 1])
        mask_recon = tf.concat(2, [mask_recon]*input_image_depth) #3)
        mask_overlap = 1 - mask_recon

        loss_recon_ori = tf.square( self.images_hiding - reconstruction )
        loss_recon_center_weight = 1 # 1/10.
        loss_recon_center = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_recon, [1,2,3]))) * loss_recon_center_weight  # Loss for non-overlapping region
        loss_recon_overlap = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_overlap, [1,2,3]))) # Loss for overlapping region
        loss_recon = loss_recon_center + loss_recon_overlap

        loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_all, labels_D))
        loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, labels_G))

        weighted_loss_adv_g = loss_adv_G * lambda_adv
        weighted_loss_recon = loss_recon * lambda_recon
        weighted_loss_adv_d = loss_adv_D * lambda_adv

        loss_G = weighted_loss_adv_g + weighted_loss_recon
        loss_D = weighted_loss_adv_d

        var_G = filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables())
        var_D = filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables())

        W_G = filter(lambda x: x.name.endswith('W:0'), var_G)
        W_D = filter(lambda x: x.name.endswith('W:0'), var_D)

        loss_G += self.config.val_weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_G)))
        loss_D += self.config.val_weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_D)))  

        #self.gradientLossBuild(reconstruction)

        # gradient reconstruction loss

        loss_recon_ori_gradient = tf.pow(tf.abs(gradient_image_hiding - gradient_reconstruction), self.config.hp_gradient_pow)
        self.gradient_loss = tf.reduce_mean(loss_recon_ori_gradient)
        weighted_gradient_loss = self.gradient_loss * self.gradient_loss_mul
        loss_G += weighted_gradient_loss

        self.g_summaryList.append(tf.summary.histogram("gradient_reconstruction_rgb", gradient_reconstruction_rgb))
        self.g_summaryList.append(tf.summary.histogram("gradient_image_hiding_rgb", gradient_image_hiding_rgb))

        self.g_summaryList.append(tf.summary.image("images_hiding", self.images_hiding))
        self.g_summaryList.append(tf.summary.image("gradient_reconstruction", gradient_reconstruction))
        self.g_summaryList.append(tf.summary.image("gradient_image_hiding", gradient_image_hiding))
        self.g_summaryList.append(tf.summary.scalar("gradient_loss", self.gradient_loss))
        self.g_summaryList.append(tf.summary.scalar("weighted_gradient_loss", weighted_gradient_loss))

        self.g_summaryList.append(tf.summary.scalar("loss_adv_G", loss_adv_G))
        self.g_summaryList.append(tf.summary.scalar("weighted_loss_adv_g", weighted_loss_adv_g))
        self.g_summaryList.append(tf.summary.scalar("loss_recon", loss_G))
        self.g_summaryList.append(tf.summary.scalar("weighted_loss_recon", weighted_loss_recon))
        self.g_summaryList.append(tf.summary.scalar("loss_G", loss_G))
        self.g_summaryList.append(tf.summary.image("G", reconstruction))

        loss_adv_D_sum = tf.summary.scalar("loss_adv_D", loss_adv_D)
        d_loss_sum = tf.summary.scalar("loss_D", loss_D)
        
        g_summary = tf.summary.merge(self.g_summaryList)
        d_summary = tf.summary.merge([loss_adv_D_sum, d_loss_sum])

        optimizer_G = tf.train.AdamOptimizer( learning_rate=self.learning_rate )
        grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
        grads_vars_G = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G)
        train_op_G = optimizer_G.apply_gradients( grads_vars_G )

        optimizer_D = tf.train.AdamOptimizer( learning_rate=self.learning_rate )
        grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
        grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
        train_op_D = optimizer_D.apply_gradients( grads_vars_D )

        self.saver = tf.train.Saver(max_to_keep=5)

        self.reconstruction = reconstruction
        self.reconstruction_ori = reconstruction_ori
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.adversarial_pos = adversarial_pos
        self.adversarial_neg = adversarial_neg
        self.adversarial_all = adversarial_all
        self.loss_recon = loss_recon
        self.loss_adv_D = loss_adv_D
        self.loss_adv_G = loss_adv_G
        self.g_summary = g_summary
        self.d_summary = d_summary
        self.train_op_G = train_op_G
        self.train_op_D = train_op_D


    def new_conv_layer( self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def channel_wise_fc_layer(self, input, name): # bottom: (7x7x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,49,49)
                    initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        # with tf.variable_scope(name):

        #     gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
        #     beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

        #     batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
        #     ema = tf.train.ExponentialMovingAverage(decay=0.5)


        #     def update():
        #         with tf.control_dependencies([ema_apply_op]):
        #             return tf.identity(batch_mean), tf.identity(batch_var)

        #     ema_apply_op = ema.apply([batch_mean, batch_var])
        #     ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        #     mean, var = tf.cond(
        #             is_train,
        #             update,
        #             lambda: (ema_mean, ema_var) )

        #     normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)

        with tf.variable_scope(name):

            normed = tf.contrib.layers.batch_norm(bottom, decay=0.5, epsilon=epsilon, scale=False)


        return normed

    def build_reconstruction( self, images, input_image_depth, ouput_image_depth, is_train ):
        # batch_size = images.get_shape().as_list()[0]

        encoderLayerNum = self.encoderLayerNum
        decoderLayerNum = self.decoderLayerNum

        with tf.variable_scope('GEN'):
            # conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            # bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            # conv2 = self.new_conv_layer(bn1, [4,4,64,64], stride=2, name="conv2" )
            # bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            # conv3 = self.new_conv_layer(bn2, [4,4,64,128], stride=2, name="conv3")
            # bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            # conv4 = self.new_conv_layer(bn3, [4,4,128,256], stride=2, name="conv4")
            # bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))
            # conv5 = self.new_conv_layer(bn4, [4,4,256,512], stride=2, name="conv5")
            # bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5'))
            # conv6 = self.new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6')
            # bn6 = self.leaky_relu(self.batchnorm(conv6, is_train, name='bn6'))

            # deconv4 = self.new_deconv_layer( bn6, [4,4,512,4000], conv5.get_shape().as_list(), padding='VALID', stride=2, name="deconv4")
            # debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train, name='debn4'))
            # deconv3 = self.new_deconv_layer( debn4, [4,4,256,512], conv4.get_shape().as_list(), stride=2, name="deconv3")
            # debn3 = tf.nn.relu(self.batchnorm(deconv3, is_train, name='debn3'))
            # deconv2 = self.new_deconv_layer( debn3, [4,4,128,256], conv3.get_shape().as_list(), stride=2, name="deconv2")
            # debn2 = tf.nn.relu(self.batchnorm(deconv2, is_train, name='debn2'))
            # deconv1 = self.new_deconv_layer( debn2, [4,4,64,128], conv2.get_shape().as_list(), stride=2, name="deconv1")
            # debn1 = tf.nn.relu(self.batchnorm(deconv1, is_train, name='debn1'))
            # recon = self.new_deconv_layer( debn1, [4,4,3,64], [batch_size,64,64,3], stride=2, name="recon")

            # encoder

            previousFeatureMap = images
            previousDepth = input_image_depth
            depth = 64

            for layer in range(1, encoderLayerNum):
                print("build_reconstruction encoder layer=", layer)
                conv = self.new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv" + str(layer)))
                bn = self.leaky_relu(self.batchnorm(conv, is_train, name=("bn" + str(layer))))
                previousFeatureMap = bn
                previousDepth = depth
                depth = depth * 2

            # last layer
            conv = self.new_conv_layer(previousFeatureMap, [4,4,previousDepth,4000], stride=2, padding='VALID', name=('conv' + str(encoderLayerNum)))
            bn = self.leaky_relu(self.batchnorm(conv, is_train, name=("bn" + str(encoderLayerNum))))

            # decoder

            previousDepth = 4000
            depth = 64 * pow(2,decoderLayerNum-2)
            featureMapSize = 4

            deconv = self.new_deconv_layer( bn, [4,4,depth,previousDepth], [self.config.size_batch,featureMapSize,featureMapSize,depth], padding='VALID', stride=2, name=("deconv" + str(decoderLayerNum)))
            debn = tf.nn.relu(self.batchnorm(deconv, is_train, name=("debn" + str(decoderLayerNum))))

            previousFeatureMap = debn

            previousDepth = depth
            depth = depth / 2
            featureMapSize = featureMapSize *2

            for layer in range(decoderLayerNum-1,1, -1):
                print("build_reconstruction decoder layer=", layer)
                deconv = self.new_deconv_layer( previousFeatureMap, [4,4,depth,previousDepth], [self.config.size_batch,featureMapSize,featureMapSize,depth], stride=2, name=("deconv" + str(layer)))
                debn = tf.nn.relu(self.batchnorm(deconv, is_train, name=('debn'+ str(layer))))
                previousFeatureMap = debn
                previousDepth = depth
                depth = depth / 2
                featureMapSize = featureMapSize *2

            recon = self.new_deconv_layer( debn, [4,4,ouput_image_depth,previousDepth], [self.config.size_batch,self.config.size_hiding,self.config.size_hiding,ouput_image_depth], stride=2, name="recon")

        # return bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, recon, tf.nn.tanh(recon)
        return recon, tf.nn.tanh(recon)

    def build_adversarial(self, images, image_depth, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            # conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            # bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            # conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2")
            # bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            # conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3")
            # bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            # conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4")
            # bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))

            # output = self.new_fc_layer( bn4, output_size=1, name='output')

            encoderLayerNum = self.encoderLayerNum

            previousFeatureMap = images
            previousDepth = image_depth
            depth = 64

            for layer in range(1, encoderLayerNum):
                print("build_adversarial encoder layer=", layer)
                conv = self.new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv" + str(layer)))
                bn = self.leaky_relu(self.batchnorm(conv, is_train, name=("bn" + str(layer))))
                previousFeatureMap = bn
                previousDepth = depth
                depth = depth * 2

            output = self.new_fc_layer( bn, output_size=1, name='output')

        return output[:,0]

    def test(self, iters):

        sess = self.sess

        trainset = self.trainset
        testset = self.testset

        samplesToShow = 3

        crop_pos = (self.config.size_image - self.config.size_hiding)/2


        # if iters == 0:
        #     self.imageSamples = tf.placeholder( tf.float32, [samplesToShow, self.config.size_image, self.config.size_image, self.input_image_depth], name="imageSamples")
        #     self.sampleImgSummary = tf.summary.image(("imageSamples (per "+str(samplePerIteration)+" iterations)"),  self.imageSamples)
        #     self.sampleImgMergeSummary = tf.summary.merge([self.sampleImgSummary])

        #print(" iters % bigSamplePerIterations=",  (iters % bigSamplePerIterations), " samplePerIteration=", samplePerIteration)

        if iters % self.config.sample_result_per_iters == 0:

            jj=0

            for start,end in zip(
                    range(0, len(testset), self.config.size_batch),
                    range(self.config.size_batch, len(testset), self.config.size_batch)):

                test_image_paths = self.testset[start:end]['image_path'].values
                test_images_ori = map(lambda x: load_image(x,pre_width=(self.config.size_image), pre_height=(self.config.size_image),width=self.config.size_image,height=self.config.size_image), test_image_paths)

                def get_test_image_name(test_image_path):
                    test_image_path = test_image_path.rsplit('/', 1)
                    test_image_path = test_image_path[len(test_image_path)-1]
                    return test_image_path

                test_images_names = map(lambda x: get_test_image_name(x), test_image_paths)

                test_images_crop = map(lambda x: crop_random(x, width=self.config.size_hiding,height=self.config.size_hiding, x=crop_pos, y=crop_pos, overlap=self.config.size_overlap), test_images_ori)
                test_images, test_crops, xs,ys = zip(*test_images_crop)

                reconstruction_vals, recon_ori_vals, loss_G_val, loss_D_val = sess.run(
                        [self.reconstruction, self.reconstruction_ori, self.loss_G, self.loss_D],
                        feed_dict={
                            self.images_tf: test_images,
                            self.images_hiding: test_crops,
                            self.is_train: False,
                            self.gradient_loss_mul: 0
                            })


                ii = 0
                for rec_val, img,x,y in zip(reconstruction_vals, test_images, xs, ys):
                    rec_hid = (255. * (rec_val+1)/2.).astype(np.uint8)
                    rec_con = (255. * (img+1)/2.).astype(np.uint8)

                    rec_con[y:y+self.config.size_hiding, x:x+self.config.size_hiding] = rec_hid                    

                    if len(rec_con.shape) >= 3:
                        rec_con = cv2.cvtColor(rec_con, cv2.COLOR_RGBA2BGRA)

                    cv2.imwrite( os.path.join(self.config.path_result, 'img_'+test_images_names[ii]+'_'+str(int(iters))+'.png'), rec_con)
                    ii += 1

                if iters == 0:
                    ii = 0
                    for test_image in test_images_ori:
                        test_image = (255. * (test_image+1)/2.).astype(np.uint8)
                        test_image[crop_pos:crop_pos+self.config.size_hiding,crop_pos:crop_pos+self.config.size_hiding] = 0

                        if len(test_image.shape) >= 3:
                            test_image = cv2.cvtColor(test_image, cv2.COLOR_RGBA2BGRA)

                        cv2.imwrite( os.path.join(self.config.path_result, 'img_'+test_images_names[ii]+'_cropped_ori.png'), test_image)
                        ii += 1

                    ii = 0
                    for test_image_path in test_image_paths:
                        
                        image = skimage.io.imread( test_image_path )

                        if len(image.shape) >= 3:
                            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

                        cv2.imwrite( os.path.join(self.config.path_result, 'img_'+test_images_names[ii]+'_ori.png'), image)
                        ii += 1

                if jj == 0:
                    print "========================================================================"
                    # print bn1_val.max(), bn1_val.min()
                    # print bn2_val.max(), bn2_val.min()
                    # print bn3_val.max(), bn3_val.min()
                    # print bn4_val.max(), bn4_val.min()
                    # print bn5_val.max(), bn5_val.min()
                    # print bn6_val.max(), bn6_val.min()
                    # print debn4_val.max(), debn4_val.min()
                    # print debn3_val.max(), debn3_val.min()
                    # print debn2_val.max(), debn2_val.min()
                    # print debn1_val.max(), debn1_val.min()
                    print recon_ori_vals.max(), recon_ori_vals.min()
                    print reconstruction_vals.max(), reconstruction_vals.min()
                    print loss_G_val, loss_D_val
                    print "========================================================================="

                    if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                        print "NaN detected!!"
                        #ipdb.set_trace()

                jj += 1


    def Run(self):

        sess = self.sess

        trainset = self.trainset
        learning_rate_val = self.config.val_learning_rate

        tf.initialize_all_variables().run()

        if self.config.path_pretrained_model is not None and os.path.exists( self.config.path_pretrained_model ):
            self.saver.restore( sess, self.config.path_pretrained_model )

        iters = 0

        loss_D_val = 0.
        loss_G_val = 0.

        crop_pos = (self.config.size_image - self.config.size_hiding)/2


        for epoch in range(self.config.val_n_epochs):
            trainset.index = range(len(trainset))
            trainset = trainset.ix[np.random.permutation(len(trainset))]

            for start,end in zip(
                    range(0, len(trainset), self.config.size_batch),
                    range(self.config.size_batch, len(trainset), self.config.size_batch)):

                image_paths = trainset[start:end]['image_path'].values
                images_ori = map(lambda x: load_image( x ,pre_width=(self.config.size_image+18), pre_height=(self.config.size_image+18),width=self.config.size_image,height=self.config.size_image), image_paths)
                is_none = np.sum(map(lambda x: x is None, images_ori))
                if is_none > 0: continue

                images_crops = map(lambda x: crop_random(x,width=self.config.size_hiding,height=self.config.size_hiding,x=crop_pos, y=crop_pos, overlap=self.config.size_overlap, isShake=True), images_ori)
                images, crops,_xs,_ys = zip(*images_crops)

                iterGradientLoss = 0.
                if iters >= self.config.gradientDelay:

                    gradient_iters_time = iters - self.config.gradientDelay

                    gradient_progress = min((gradient_iters_time / self.config.adaptive_lambda_gradient_loss_grow_time_iter ), 1)

                    iterGradientLoss = ((self.config.lambda_gradient_loss - self.config.lambda_starting_gradient_loss) * gradient_progress) + self.config.lambda_starting_gradient_loss

                # Printing activations every 100 iterations
                if iters % self.config.sample_result_per_iters == 0:
                    self.test(iters)

                # Generative Part is updated every iteration

                _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals, g_summary_str = sess.run(
                        [self.train_op_G, self.loss_G, self.adversarial_pos, 
                        self.adversarial_neg, self.loss_recon, self.loss_adv_G, 
                        self.reconstruction, self.reconstruction_ori,self.g_summary],
                        feed_dict={
                            self.images_tf: images,
                            self.images_hiding: crops,
                            self.learning_rate: learning_rate_val,
                            self.is_train: True,
                            self.gradient_loss_mul: iterGradientLoss
                            })

                self.writer.add_summary(g_summary_str, iters)

                _, loss_D_val, adv_pos_val, adv_neg_val, d_summary_str = sess.run(
                        [self.train_op_D, self.loss_D, self.adversarial_pos, 
                        self.adversarial_neg,self.d_summary],
                        feed_dict={
                            self.images_tf: images,
                            self.images_hiding: crops,
                            self.learning_rate: learning_rate_val/10.,
                            self.is_train: True,
                            self.gradient_loss_mul: iterGradientLoss
                                })

                self.writer.add_summary(d_summary_str, iters)

                print "Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_G_val,  "Dis Loss:", loss_D_val, "||||", adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max()

                iters += 1


            self.saver.save(self.sess, self.config.path_model + 'model', global_step=epoch)
            learning_rate_val *= 0.99






