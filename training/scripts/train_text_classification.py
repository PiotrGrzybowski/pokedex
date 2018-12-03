import numpy as np
import argparse

import tensorflow as tf
import os

from tensorflow.contrib import rnn
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer

from data.tf_recorfd import build_inputs_from_cifar_tf_record_data, build_inputs_from_numpy_text_names_data
from training.image_classification import build_model_specification, build_eval_model_specification, train
from training.scripts.evaluate import evaluate


def build_rnn(samples):
    num_input = 27  # MNIST data input (img shape: 28*28)
    timesteps = 11  # timesteps
    num_hidden = 100  # hidden layer num of features
    num_classes = 2  # MNIST total classes (0-9 digits)

    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    x = tf.transpose(samples, [0, 2, 1])
    x = tf.unstack(samples, timesteps, 1)
    out = samples
    with tf.variable_scope('rnn'):
        lstm_cell = tf.nn.rnn_cell.BasicRNNCell(100)

        output, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        out = tf.matmul(output[-1], weights['out']) + biases['out']

    with tf.variable_scope('fc1'):
        out = tf.layers.dense(inputs=out, units=2, activation=tf.nn.softmax)

    return out


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", required=True)
parser.add_argument("--learning_rate", required=True)

args = vars(parser.parse_args())

train_path = '/home/piotr/Workspace/Projects/pokedex/data/train.npz'
test_path = '/home/piotr/Workspace/Projects/pokedex/data/test.npz'

batch_size = int(args["batch_size"])

train_inputs = build_inputs_from_numpy_text_names_data(train_path, batch_size)
test_inputs = build_inputs_from_numpy_text_names_data(test_path, batch_size)

# print(train_inputs)
# variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
# with tf.Session() as sess:
#     sess.run(variable_init_op)
#     sess.run(train_inputs['iterator_init_op'])
#
#     print(sess.run(train_inputs['labels']))

images = train_inputs['images']
labels = train_inputs['labels']

TRAIN_SIZE = 22467
EVAL_SIZE = 5617

train_steps = TRAIN_SIZE // batch_size
test_steps = EVAL_SIZE // batch_size

learning_rate = float(args["learning_rate"])

reg = 'l2-0.1'
opt = 'Adam'
#model_dir = '/DATA/piotr/cifar/learning/{} op: {} lr: {} batch: {:03d} reg: {}'.format([500], opt, learning_rate, batch_size, reg)
model_dir = '/DATA/piotr/cifar/sprawko/88888'
# model_dir = '/DATA/piotr/cifar/learning/apaktest'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

loss_fn = tf.losses.sparse_softmax_cross_entropy
optimizer = tf.train.AdamOptimizer()
model_fn = build_rnn
train_spec = build_model_specification(train_inputs, model_fn, loss_fn, optimizer)
eval_spec = build_eval_model_specification(test_inputs, model_fn, loss_fn)
train(train_spec, eval_spec, model_dir, 30, train_steps, test_steps)

# model_dir = '/DATA/piotr/cifar/learning/best'
# evaluate(eval_spec, model_dir)