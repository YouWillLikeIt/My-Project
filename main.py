import tensorflow as tf
import os
from model import MODEL

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_boolean("is_eval", False, "if the eval")
flags.DEFINE_string("model_dir", 'Model', "the dir of model")
flags.DEFINE_string("checkpoint_dir", '20181120', "the checkpoint of model")

flags.DEFINE_integer("block_x2", 4, "the num of 64block")
flags.DEFINE_integer("block_x3", 4, "the num of 96block")
flags.DEFINE_integer("block_x4", 4, "the num of 128block")

flags.DEFINE_float("learning_rate", 0.01, "the learning rate")  ###################
flags.DEFINE_integer("image_size", 32, "the size of image input")
flags.DEFINE_integer("batch_size", 64, "the size of batch")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")
flags.DEFINE_integer("out_channel", 64, "the channel of output")
flags.DEFINE_string("gpu_device", '1', "the number of gpu using")
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device


def main(_):
    model = MODEL(tf.Session(),
                  is_train=FLAGS.is_train,
                  is_eval=FLAGS.is_eval,
                  image_size=FLAGS.image_size,
                  batch_size=FLAGS.batch_size,
                  kernel_size=FLAGS.kernel_size,
                  out_channel=FLAGS.out_channel,
                  block_x2=FLAGS.block_x2,
                  block_x3=FLAGS.block_x3,
                  block_x4=FLAGS.block_x4)
    if model.is_train:
        model.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
