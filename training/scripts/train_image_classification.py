import numpy as np
import argparse

import tensorflow as tf
import os

from tensorflow.contrib.layers import l2_regularizer, xavier_initializer

from data.tf_recorfd import build_inputs_from_cifar_tf_record_data
from training.image_classification import build_model_specification, build_eval_model_specification, train
from training.scripts.evaluate import evaluate


def basic_model(samples):
    out = samples

    with tf.variable_scope('conv1'):
        out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[3, 3], strides=2, padding='SAME')

    with tf.variable_scope('conv2'):
        out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[3, 3], strides=2, padding='SAME')

    with tf.variable_scope('fc'):
        out = tf.reshape(out, [-1, 8 * 8 * 64])
        out = tf.layers.dense(inputs=out, units=384, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=192, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=10, activation=tf.nn.softmax)

    return out


def build_model(samples):
    out = samples

    with tf.variable_scope('conv1'):
        out = tf.layers.conv2d(inputs=out, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, padding='SAME')
        out = tf.layers.dropout(inputs=out, rate=0.25)

    with tf.variable_scope('conv2'):
        out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, padding='SAME')
        out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[2, 2], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, padding='SAME')
        out = tf.layers.dropout(inputs=out, rate=0.25)

    with tf.variable_scope('fc1'):
        out = tf.reshape(out, [-1, 4 * 4 * 128])
        out = tf.layers.dense(inputs=out, units=1500, activation=tf.nn.relu)
        out = tf.layers.dropout(inputs=out, rate=0.5)
        out = tf.layers.dense(inputs=out, units=10, activation=tf.nn.softmax)

    return out

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", required=True)
parser.add_argument("--learning_rate", required=True)

args = vars(parser.parse_args())

train_path = '/home/piotr/Workspace/Repositories/models/tutorials/image/cifar10_estimator/cifar-10-data/train.tfrecords'
test_path = '/home/piotr/Workspace/Repositories/models/tutorials/image/cifar10_estimator/cifar-10-data/eval.tfrecords'

batch_size = int(args["batch_size"])

train_inputs = build_inputs_from_cifar_tf_record_data(train_path, batch_size, 8, True)
test_inputs = build_inputs_from_cifar_tf_record_data(test_path, batch_size, 8)

images = train_inputs['images']
labels = train_inputs['labels']

TRAIN_SIZE = 50000
EVAL_SIZE = 10000

train_steps = TRAIN_SIZE // batch_size
test_steps = EVAL_SIZE // batch_size

learning_rate = float(args["learning_rate"])

reg = 'l2-0.1'
opt = 'Adam'
#model_dir = '/DATA/piotr/cifar/learning/{} op: {} lr: {} batch: {:03d} reg: {}'.format([500], opt, learning_rate, batch_size, reg)
model_dir = '/DATA/piotr/cifar/sprawko/7'
# model_dir = '/DATA/piotr/cifar/learning/apaktest'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

loss_fn = tf.losses.sparse_softmax_cross_entropy
optimizer = tf.train.AdamOptimizer()
model_fn = basic_model
train_spec = build_model_specification(train_inputs, model_fn, loss_fn, optimizer)
eval_spec = build_eval_model_specification(test_inputs, model_fn, loss_fn)
train(train_spec, eval_spec, model_dir, 30, train_steps, test_steps)

# model_dir = '/DATA/piotr/cifar/learning/best'
# evaluate(eval_spec, model_dir)