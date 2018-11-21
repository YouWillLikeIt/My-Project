import tensorflow as tf
import time
import os
import numpy as np


class MODEL(object):
    def __init__(self, sess, is_train, is_eval, image_size, batch_size, kernel_size, out_channel, block_x2, block_x3,
                 block_x4):
        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.block_x2 = block_x2
        self.block_x3 = block_x3
        self.block_x4 = block_x4
        self.in_dim = 3

        # self.images = None
        # self.labels_x2 = None
        # self.labels_x3 = None
        # self.labels_x4 = None

    def train(self, config):
        print('start training')

        image_shape = [None, self.image_size, self.image_size, self.in_dim]
        label_x2_shape = [None, self.image_size * 2, self.image_size * 2, self.in_dim]
        label_x3_shape = [None, self.image_size * 3, self.image_size * 3, self.in_dim]
        label_x4_shape = [None, self.image_size * 4, self.image_size * 4, self.in_dim]
        self.build_model(image_shape, labels_shape=[label_x2_shape, label_x3_shape, label_x4_shape])
        train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(
            self.loss)  ###########################
        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(config.model_dir, config.chechpoint_dir), self.sess.graph)

        counter = self.load(config.model_dir)  ###################
        time_ = time.time()
        print("\nNow Start Training...\n")
        # for ep in range(200):
        #     pass

    def Params32(self):
        out_channel = self.out_channel  # 64
        ks = self.kernel_size
        weights = {
            'w_32_1': tf.Variable(tf.random_normal([ks, ks, self.in_dim, out_channel], stddev=0.01), name='w_32_1'),
            'w_32_2': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_32_2'),
            'w_32_3': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_32_3')
        }
        bias = {
            'b_32_1': tf.Variable(tf.zeros([out_channel], name='b_32_1')),
            'b_32_2': tf.Variable(tf.zeros([out_channel], name='b_32_2')),
            'b_32_3': tf.Variable(tf.zeros([out_channel], name='b_32_3')),
        }
        return weights, bias

    def Params64(self):
        out_channel = self.out_channel  # 64
        ks = self.kernel_size
        weights = {
            'w_64_1': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_64_1'),
            'w_64_2': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_64_2'),
            'w_64_3': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_64_3')
        }
        bias = {
            'b_64_1': tf.Variable(tf.zeros([out_channel], name='b_64_1')),
            'b_64_2': tf.Variable(tf.zeros([out_channel], name='b_64_2')),
            'b_64_3': tf.Variable(tf.zeros([out_channel], name='b_64_3')),
        }
        return weights, bias

    def Params_resize(self):
        out_channel = self.out_channel  # 64
        ks = self.kernel_size
        weights = {
            'w_32_64': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_32_64'),
            'w_32_96': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_32_96'),
            'w_64_96': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_64_96'),
            'w_96_64': tf.Variable(tf.random_normal([ks, ks, 3, out_channel], stddev=0.01), name='w_96_64'),

        }
        bias = {
            'b_32_64': tf.Variable(tf.zeros([out_channel], name='b_32_64')),
            'b_32_96': tf.Variable(tf.zeros([out_channel], name='b_32_96')),
            'b_64_96': tf.Variable(tf.zeros([out_channel], name='b_64_96')),
            'b_96_64': tf.Variable(tf.zeros([out_channel], name='b_96_64')),
        }
        return weights, bias

    def Params_down_64_32(self):
        out_channel = self.out_channel  # 64
        ks = self.kernel_size
        weights = {
            'w_1': tf.Variable(tf.random_normal([ks, ks, self.in_dim, out_channel], stddev=0.01), name='w_1'),
            'w_2': tf.Variable(tf.random_normal([ks, ks, self.out_channel, out_channel], stddev=0.01), name='w_2'),
            'w_3': tf.Variable(tf.random_normal([ks, ks, self.out_channel, out_channel], stddev=0.01), name='w_3'),
        }
        bias = {
            'b_1': tf.Variable(tf.zeros([out_channel], name='b_1')),
            'b_2': tf.Variable(tf.zeros([out_channel], name='b_2')),
            'b_3': tf.Variable(tf.zeros([out_channel], name='b_3')),
        }
        return weights, bias

    def Params96(self):
        out_channel = self.out_channel  # 64
        ks = self.kernel_size
        weights = {
            'w_96_1': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_96_1'),
            'w_96_2': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_96_2'),
            'w_96_3': tf.Variable(tf.random_normal([ks, ks, out_channel, out_channel], stddev=0.01), name='w_96_3')
        }
        bias = {
            'b_96_1': tf.Variable(tf.zeros([out_channel], name='b_96_1')),
            'b_96_2': tf.Variable(tf.zeros([out_channel], name='b_96_2')),
            'b_96_3': tf.Variable(tf.zeros([out_channel], name='b_96_3')),
        }
        return weights, bias

    def Params_final(self):
        out_channel = self.out_channel  # 64
        ks = self.kernel_size
        weights = {
            'w_64': tf.Variable(tf.random_normal([ks, ks, out_channel, 3], stddev=0.01), name='w_64'),
            'w_96': tf.Variable(tf.random_normal([ks, ks, out_channel, 3], stddev=0.01), name='w_96'),
            'w_128': tf.Variable(tf.random_normal([ks, ks, out_channel, 3], stddev=0.01), name='w_128')
        }
        bias = {
            'b_64': tf.Variable(tf.zeros([3], name='b_64')),
            'b_96': tf.Variable(tf.zeros([3], name='b_96')),
            'b_128': tf.Variable(tf.zeros([3], name='b_128')),
        }
        return weights, bias

    def build_model(self, images_shape, labels_shape):
        images = tf.placeholder(tf.float32, images_shape)
        labels_x2 = tf.placeholder(tf.float32, labels_shape[0])
        labels_x3 = tf.placeholder(tf.float32, labels_shape[1])
        labels_x4 = tf.placeholder(tf.float32, labels_shape[2])
        self.params32, self.bias32 = self.Params32()
        self.params64, self.bias64 = self.Params64()
        self.params_down_64_32, self.bias_down_64_32 = self.Params_down_64_32()
        self.params96, self.bias96 = self.Params96()

        self.params_resize, self.bias_resize = self.Params_resize()
        self.params_final, self.bias_final = self.Params_final()
        # self.params64_final = tf.Variable(
        #     tf.random_normal([self.kernel_size, self.kernel_size, self.out_channel, self.in_dim], stddev=0.01),
        #     name='w64_f')
        # self.params64_final = tf.Variable(tf.zeros([self.in_dim], name='b64_f'))
        # self/\
        self.model()

    def model(self):
        pro_32 = self.process32(self.image_size)
        pro_64 = self.block64(pro_32)

        resize_32_64 = self.resize32_64(pro_32)
        result_64 = tf.nn.conv2d(pro_64 + resize_32_64, self.params_final['w_64'], strides=[1, 1, 1, 1],
                                 padding='SAME') + self.bias_final['b_64']

        down_64_32 = self.down64_32(pro_64)
        pro_96 = self.block96(down_64_32, pro_64)
        result_96 = tf.nn.conv2d(pro_96, self.params_final['w_96'], strides=[1, 1, 1, 1], padding='SAME') + \
                    self.bias_final['b_96']

        # resize_64_32 = self.process32(result_64, down_sample=True)
        # pro32 = self.process32()

    def process32(self, input_layer):
        temp = input_layer
        out = input_layer
        out = tf.nn.conv2d(out, self.params32['w_32_1'], strides=[1, 1, 1, 1], padding='SAME') + self.bias32['b_32_1']
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, self.params32['w_32_2'], strides=[1, 1, 1, 1], padding='SAME') + self.bias32['b_32_2']
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, self.params32['w_32_3'], strides=[1, 1, 1, 1], padding='SAME') + self.bias32['b_32_3']
        out = out + temp
        out = tf.nn.relu(out)
        return out

    def block64(self, input_layer):
        # temp = input_layer
        out = tf.nn.conv2d_transpose(input_layer, self.params64['w_64_1'],
                                     output_shape=[self.batch_size, 64, 64, self.out_channel],
                                     strides=[1, 2, 2, 1], padding='SAME') + self.bias64['b_64_1']
        out = input_layer
        for block in range(0, self.block_x2):
            temp = out
            if block == 0:
                out = tf.nn.conv2d_transpose(out, self.params64['w_64_1'],
                                             output_shape=[self.batch_size, 64, 64, self.out_channel],
                                             strides=[1, 2, 2, 1], padding='SAME') + self.bias64['b_64_1']
            else:
                out = tf.nn.conv2d(out, self.params64['w_64_1'], strides=[1, 1, 1, 1], padding='SAME') + self.bias64[
                    'b_64_1']
            out = tf.nn.relu(out)
            out = tf.nn.conv2d(out, self.params64['w_64_2'], strides=[1, 1, 1, 1], padding='SAME') + self.bias64[
                'b_64_2']
            out = tf.nn.relu(out)
            out = tf.nn.conv2d(out, self.params64['w_64_3'], strides=[1, 1, 1, 1], padding='SAME') + self.bias64[
                'b_64_3']
            out = out + temp
            out = tf.nn.relu(out)
        return out

    # def resize32_64(self, input_layer):
    #     return tf.nn.conv2d_transpose(input_layer, self.params_resize['w_32_64'],
    #                                   output_shape=[self.batch_size, 64, 64, self.out_channel], strides=[1, 2, 2, 1],
    #                                   padding='SAME') + self.params_resize['b_32_64']

    def down64_32(self, input_layer):
        temp = input_layer
        out = tf.nn.conv2d(input_layer, self.params_down_64_32['w_1'], strides=[1, 2, 2, 1], padding='SAME') + \
              self.bias_down_64_32['b_1']
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, self.params_down_64_32['w_2'], strides=[1, 1, 1, 1], padding='SAME') + \
              self.bias_down_64_32['b_2']
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, self.params_down_64_32['w_3'], strides=[1, 1, 1, 1], padding='SAME') + \
              self.bias_down_64_32['b_3']
        out = out + temp
        out = tf.nn.relu(out)
        return out

    def block96(self, in_32, in_64):
        resize32 = tf.nn.conv2d_transpose(in_32, output_shape=[self.batch_size, 96, 96, self.out_channel],
                                          filter=self.params_resize['w_32_96'], strides=[1, 1, 1, 1], padding='SAME') + \
                   self.params_resize['b_32_96']
        resize64 = tf.nn.conv2d_transpose(in_64, output_shape=[self.batch_size, 96, 96, self.out_channel],
                                          filter=self.params_resize['w_64_96'], strides=[1, 1, 1, 1], padding='SAME') + \
                   self.params_resize['b_64_96']
        out = resize32 + resize64
        for block in range(self.block_x3):
            temp = out
            out = tf.nn.conv2d(out, self.params96['w_96_1'], strides=[1, 1, 1, 1], padding='SAME') + self.bias96[
                'b_96_1']
            out = tf.nn.relu(out)
            out = tf.nn.conv2d(out, self.params96['w_96_2'], strides=[1, 1, 1, 1], padding='SAME') + self.bias96[
                'b_96_2']
            out = tf.nn.relu(out)
            out = tf.nn.conv2d(out, self.params96['w_96_3'], strides=[1, 1, 1, 1], padding='SAME') + self.bias96[
                'b_96_3']
            out = tf.nn.relu(temp + out)
        return out

    def down96_64(self, input_layer):
        out = tf.nn.conv2d(input_layer, filter=self.params_resize['w_96_64'], strides=[1, 3, 3, 1], padding='SAME')
