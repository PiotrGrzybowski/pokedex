import functools

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16

from data.tf_recorfd import build_inputs_from_tf_record_data, build_inputs_from_cifar_tf_record_data
from training.image_classification import build_model_specification, build_eval_model_specification, train

train_path = '/DATA/piotr/cifar/data/processed/cifar-train.record'
test_path = '/DATA/piotr/cifar/data/processed/cifar-eval.record'
model_dir = '/DATA/piotr/cifar/learning/logs'

batch_size = 32


train_inputs = build_inputs_from_cifar_tf_record_data(train_path, batch_size, 8)
test_inputs = build_inputs_from_cifar_tf_record_data(test_path, batch_size, 8)


#with tf.Session() as sess:
#    sess.run(train_inputs['iterator_init_op'])

    #print(sess.run(train_inputs["images"]).shape)
    #print(sess.run(train_inputs["labels"]).shape)



TRAIN_SIZE = 50000
EVAL_SIZE = 10000

train_steps = TRAIN_SIZE // batch_size
test_steps = EVAL_SIZE // batch_size


def vgg(inputs):
    net, end_points = vgg_16(inputs, num_classes=10)
    return net


def build_model(samples):
    out = samples

    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 2000)
        out = tf.nn.relu(out)

    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, 50)
        out = tf.nn.relu(out)
        # out = tf.nn.dropout(out, 0.2)

    with tf.variable_scope('fc_3'):
        out = tf.layers.dense(out, 10)

    return out


# def build_model(images):
#     """Compute logits of the model (output distribution)
#
#     Args:
#         is_training: (bool) whether we are training or not
#         inputs: (dict) contains the inputs of the graph (features, labels...)
#                 this can be `tf.placeholder` or outputs of `tf.data`
#         params: (Params) hyperparameters
#
#     Returns:
#         output: (tf.Tensor) output of the model
#     """
#     # assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
#
#     out = images
#     # Define the number of channels of each convolution
#     # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
#     num_channels = 3
#     channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
#     for i, c in enumerate(channels):
#         with tf.variable_scope('block_{}'.format(i+1)):
#             out = tf.layers.conv2d(out, c, 3, padding='same')
#             out = tf.nn.relu(out)
#             out = tf.layers.max_pooling2d(out, 2, 2)
#
#     assert out.get_shape().as_list() == [None, 4, 4, num_channels * 8]
#
#     out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
#     with tf.variable_scope('fc_1'):
#         out = tf.layers.dense(out, num_channels * 8)
#         out = tf.nn.relu(out)
#     with tf.variable_scope('fc_2'):
#         logits = tf.layers.dense(out, 10)
#
#     return logits


loss_fn = tf.losses.sparse_softmax_cross_entropy
optimizer = tf.train.MomentumOptimizer(0.0001, 0.9)
model_fn = build_model
train_spec = build_model_specification(train_inputs, model_fn, loss_fn, optimizer)
eval_spec = build_eval_model_specification(test_inputs, model_fn, loss_fn)
train(train_spec, eval_spec, model_dir, 1000, train_steps, test_steps)