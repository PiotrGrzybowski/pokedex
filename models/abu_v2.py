import tensorflow as tf
from tensorflow.python import GraphKeys

from datasets.mnist import paget_mlp

destination = '/Users/Piotr/Workspace/DataScience/pokedex/datasets'
x_train, y_train, init_train, x_test, y_test, init_test = paget_mlp(destination, 100, prefetch=2, cores=4)

if __name__ == '__main__':
    w1 = tf.Variable(tf.random_normal(shape=[784, 100], stddev=0.05, dtype=tf.float32),
                     collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.WEIGHTS],
                     name='w1')

    b1 = tf.Variable(tf.zeros(shape=[1, 100]),
                     collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.BIASES],
                     name='b1')

    w2 = tf.Variable(tf.random_normal(shape=[100, 10], stddev=0.05, dtype=tf.float32),
                     collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.WEIGHTS],
                     name='w2')

    b2 = tf.Variable(tf.zeros(shape=[1, 10]),
                     collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.BIASES],
                     name='b2')

    weights = tf.get_collection('weights')
    biases = tf.get_collection('biases')
    new_variables = weights + biases

    # input_data = tf.placeholder(shape=[None, 784], name='input_data', dtype=tf.float32)
    # labels = tf.placeholder(shape=[None, 10], name='labels', dtype=tf.float32)

    input_data = x_train
    labels = y_train

    out = tf.matmul(input_data, w1)
    out = tf.add(out, b1)
    out = tf.nn.relu(out)

    out = tf.matmul(out, w2)
    out = tf.add(out, b2)
    out = tf.nn.softmax(out)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=labels))
    optimizer = tf.train.AdamOptimizer()
    gradients = optimizer.compute_gradients(loss=loss, var_list=new_variables)
    train_step = optimizer.apply_gradients(gradients)

    tf_accuracy_metric, tf_accuracy_update = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(out, 1), name='accuracy_metric')
    accuracy_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")
    # tf_accuracy_initializer = tf.variables_initializer(accuracy_vars)

    tf_loss_metric, tf_loss_update = tf.metrics.mean(loss, name='loss_metric')
    loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="loss_metric")
    init_new_vars_op = tf.variables_initializer(new_variables)
    init_metric_vars_op = tf.variables_initializer(loss_vars + accuracy_vars)
    batch_size = 100

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(out, 1))
        with tf.name_scope("batch_accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope("epoch_accuracy"):
            epoch_accuracy = tf.metrics.mean(accuracy)

    with tf.Session() as sess:
        sess.run(init_new_vars_op)
        sess.run(tf.variables_initializer(optimizer.variables()))

        for epoch in range(10):
            sess.run(init_metric_vars_op)
            sess.run([init_train, init_test])
            # sess.run(tf_accuracy_initializer)

            for i in range(0, 60000, batch_size):
                sess.run([train_step, tf_accuracy_update, tf_loss_update])
                # print("Accuracy at step %d: %s" % (i, sess.run(tf_accuracy_metric)))

            # print(sess.run(loss_metric[0]))
            print("Accuracy, Loss at epoch %d: %s, %s" % (epoch, sess.run(tf_accuracy_metric), sess.run(tf_loss_metric)))
