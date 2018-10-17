import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import functools
import os

from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16

from models.iris2 import build_inputs, build_model_specification, build_model, train, eval_specification

if __name__ == '__main__':
    data = load_iris()

    features = data['data']
    labels = data['target']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.3)

    inputs = build_inputs(X_train, y_train, 5)
    steps = int(X_train.shape[0] / 5)

    inputs_eval = build_inputs(X_test, y_test, 5)
    steps_eval = int(X_test.shape[0] / 5)

    loss_fn = tf.losses.sparse_softmax_cross_entropy
    optimizer = tf.train.AdamOptimizer()

    model_fn = functools.partial(vgg_16, num_classes=5)

    train_specification = build_model_specification(inputs, model_fn, loss_fn, optimizer)
    eval_spec = eval_specification(inputs, model_fn, loss_fn)

    train(train_specification, eval_spec, model_dir, 30, steps, steps_eval)
