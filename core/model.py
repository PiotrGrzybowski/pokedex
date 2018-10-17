import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

a = np.random.uniform(0, 1, (1, 28, 28, 1))

inp = tf.placeholder(shape=[1, 28, 28, 1], dtype=tf.float32, name="input")
best = tf.reduce_sum(inp)

with tf.Session() as session:
    print(session.run(best, feed_dict={inp: a}))

x = tf.constant(1.0, dtype=tf.float32, name="my-node-x")
print(x)

with tf.Session() as sess:
    print(sess.run(x))

with tf.variable_scope('model'):
    x1 = tf.get_variable('x', [], dtype=tf.float32)
    print(x1)

with tf.variable_scope('model', reuse=True):
    x2 = tf.get_variable('x', [], dtype=tf.float32)
    print(x2)

