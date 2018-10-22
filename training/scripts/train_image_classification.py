import argparse

import tensorflow as tf
import os
from data.tf_recorfd import build_inputs_from_cifar_tf_record_data
from training.image_classification import build_model_specification, build_eval_model_specification, train
from training.scripts.evaluate import evaluate


def build_model(samples):
    out = samples

    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 500, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    #with tf.variable_scope('fc_2'):
     #   out = tf.layers.dense(out, 100, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    # with tf.variable_scope('fc_3'):
    #     out = tf.layers.dense(out, 50, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    with tf.variable_scope('fc_4'):
        out = tf.layers.dense(out, 10)

    return out


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", required=True)
parser.add_argument("--learning_rate", required=True)

args = vars(parser.parse_args())

train_path = '/DATA/piotr/cifar/data/processed/cifar-train.record'
test_path = '/DATA/piotr/cifar/data/processed/cifar-eval.record'

batch_size = int(args["batch_size"])

train_inputs = build_inputs_from_cifar_tf_record_data(train_path, batch_size, 8)
test_inputs = build_inputs_from_cifar_tf_record_data(test_path, batch_size, 8)

TRAIN_SIZE = 50000
EVAL_SIZE = 10000

train_steps = TRAIN_SIZE // batch_size
test_steps = EVAL_SIZE // batch_size

learning_rate = float(args["learning_rate"])
reg = 'l2-0.1'
opt = 'Adam'
model_dir = '/DATA/piotr/cifar/learning/{} op: {} lr: {} batch: {:03d} reg: {}'.format([500], opt, learning_rate, batch_size, reg)
model_dir = '/DATA/piotr/cifar/learning/apaktest'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

loss_fn = tf.losses.sparse_softmax_cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
model_fn = build_model
train_spec = build_model_specification(train_inputs, model_fn, loss_fn, optimizer)
eval_spec = build_eval_model_specification(test_inputs, model_fn, loss_fn)
train(train_spec, eval_spec, model_dir, 2, train_steps, test_steps)

# model_dir = '/DATA/piotr/cifar/learning/best'
# evaluate(eval_spec, model_dir)