import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python import GraphKeys
from tensorflow.python import debug as tf_debug

from datasets.mnist import paget_mlp, paget_cnn

destination = '/Users/Piotr/Workspace/DataScience/pokedex/datasets'
# destination = '/home/piotr/Workspace/Projects/pokedex/datasets'
x_train, y_train, init_train, x_test, y_test, init_test = paget_cnn(destination, 100, prefetch=2, cores=4)


def random_normal_weights(stddev, shape, name):
    return tf.get_variable(name=name, dtype=tf.float32, shape=shape,
                           initializer=tf.initializers.random_normal(stddev=stddev),
                           collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.WEIGHTS])


def glorot_uniform_weights(shape, name):
    return tf.get_variable(name=name, dtype=tf.float32, shape=shape,
                           initializer=tf.initializers.glorot_uniform(),
                           collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.WEIGHTS])


def bias_zero_variable(shape, name):
    return tf.get_variable(name=name, dtype=tf.float32, shape=shape,
                           initializer=tf.initializers.zeros(),
                           collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.BIASES])


def simple_mlp(inputs):
    with tf.variable_scope('fc1'):
        w1 = random_normal_weights(0.05, [784, 100], 'w1')
        b1 = bias_zero_variable([1, 100], 'b1')
        out = tf.nn.relu(tf.add(tf.matmul(inputs, w1), b1))

    with tf.variable_scope('fc2'):
        w2 = random_normal_weights(0.05, [100, 10], 'w2')
        b2 = bias_zero_variable([1, 10], 'b2')
        out = tf.nn.softmax(tf.add(tf.matmul(out, w2), b2))

    return out


def conv2d(input, kernel_shape, strides, name, padding, activation=tf.nn.relu):
    with tf.variable_scope(name):
        weights = random_normal_weights(stddev=0.001, shape=kernel_shape, name='weights')
        bias = bias_zero_variable(shape=kernel_shape[-1], name='bias')
        out = tf.nn.conv2d(input=input, filter=weights, strides=strides, padding=padding)
        out = tf.nn.bias_add(value=out, bias=bias)
        out = activation(out)

    return out


def dense(input, units, name, activation):
    with tf.variable_scope(name):
        weights = glorot_uniform_weights(shape=[input.get_shape().as_list()[-1], units], name='weights')
        bias = bias_zero_variable(shape=[units], name='bias')
        out = activation(tf.add(tf.matmul(input, weights), bias))

    return out


def max_pool(input, pool_size, strides, name):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(value=input, ksize=pool_size, strides=strides, padding='SAME')

    return out


def simple_cnn(inputs):
    out = inputs

    out = conv2d(input=out, kernel_shape=[3, 3, 1, 64], strides=[1, 1, 1, 1], name='conv1_1', padding='SAME')
    out = max_pool(input=out, pool_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')
    #
    out = conv2d(input=out, kernel_shape=[3, 3, 64, 64], strides=[1, 1, 1, 1], name='conv2_1', padding='SAME')
    out = max_pool(input=out, pool_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    out = conv2d(input=out, kernel_shape=[7, 7, 64, 384], strides=[1, 1, 1, 1], name='fc1', padding='VALID')
    out = conv2d(input=out, kernel_shape=[1, 1, 384, 192], strides=[1, 1, 1, 1], name='fc2', padding='VALID')
    out = conv2d(input=out, kernel_shape=[1, 1, 192, 10], strides=[1, 1, 1, 1], name='fc3', padding='VALID', activation=tf.nn.softmax)
    out = tf.reshape(out, [-1, 10])

    # out = tf.reshape(out, [-1, 7*7*64])
    # out = tf.reshape(out, [-1, 192])
    # out = dense(out, 392, 'fc1', tf.nn.relu)
    # out = dense(out, 192, 'fc2', tf.nn.relu)
    # out = dense(out, 10, 'fc3', tf.nn.softmax)
    #
    # out = conv2d(input=out, kernel_shape=[3, 3, 128, 256], strides=[1, 1, 1, 1], name='conv3_1', padding='SAME')
    # out = conv2d(input=out, kernel_shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], name='conv3_2', padding='SAME')
    # out = conv2d(input=out, kernel_shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], name='conv3_3', padding='SAME')
    # out = max_pool(input=out, pool_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool3')
    #
    # out = conv2d(input=out, kernel_shape=[4, 4, 256, 4096], strides=[1, 1, 1, 1], name='fc1', padding='VALID')
    # out = conv2d(input=out, kernel_shape=[1, 1, 4096, 1000], strides=[1, 1, 1, 1], name='fc2', padding='VALID')
    # out = conv2d(input=out, kernel_shape=[1, 1, 1000, 10], strides=[1, 1, 1, 1], name='fc3', padding='VALID', activation=tf.nn.softmax)
    #
    # with tf.variable_scope('flatten'):
    return out


def basic_model(samples):
    out = samples

    with tf.variable_scope('conv1'):
        out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[3, 3], strides=2, padding='SAME')

    with tf.variable_scope('conv2'):
        out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(inputs=out, pool_size=[3, 3], strides=2, padding='SAME')

    with tf.variable_scope('fc'):
        out = tf.reshape(out, [-1, 7 * 7 * 64])
        out = tf.layers.dense(inputs=out, units=384, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=192, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=10, activation=tf.nn.softmax)

    return out


def create_metrics(labels, logits):
    metrics = {'accuracy': tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(logits, 1)),
               # 'precision': tf.metrics.precision(tf.argmax(labels, 1), tf.argmax(logits, 1)),
               # 'recall': tf.metrics.recall(tf.argmax(labels, 1), tf.argmax(logits, 1)),
               'loss': tf.metrics.mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))}

    metric_values = {name: metric[0] for name, metric in metrics.items()}
    metric_updates = tf.group(*[metric[1] for name, metric in metrics.items()])

    return metric_values, metric_updates


def variables_initializer(optimizer):
    return tf.variables_initializer(tf.get_collection('weights') +
                                    tf.get_collection('biases') +
                                    optimizer.variables())


def metrics_initializer():
    return tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics'))


if __name__ == '__main__':
    input_data = x_train
    labels = y_train

    with tf.variable_scope('model') as scope:
        out = simple_cnn(input_data)
        # out = basic_model(input_data)

    # with tf.variable_scope(scope, reuse=True):
    #     out2 = simple_cnn(x_test)

    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=labels))
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(cost_function)

    with tf.variable_scope('metrics'):
        train_metric_values, train_metric_updates = create_metrics(labels, out)
    #     test_metric_values, test_metric_updates = create_metrics(y_test, out2)

    batch_size = 100
    init_variables = variables_initializer(optimizer)
    init_metrics = metrics_initializer()

    # with tf_debug.LocalCLIDebugWrapperSession(tf.Session(), ui_type="curses") as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_variables)

        for epoch in range(10):
            sess.run([init_train, init_test, init_metrics])
            sess.run([init_train, init_test])
            for i in range(0, 60000, batch_size):
                sess.run([train_step, train_metric_updates])
                print(sess.run(train_metric_values))
            # print(epoch)
            print(sess.run(train_metric_values))
            #
            # for i in range(0, 10000, batch_size):
            #     sess.run([test_metric_updates])
            # print(sess.run(test_metric_values))
