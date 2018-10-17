import json
import os

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()

features = data['data']
labels = data['target']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.3)


def build_inputs(features, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    irises, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    return {'samples': irises, 'labels': labels, 'iterator_init_op': iterator_init_op}


def build_model_savers(model_dir):
    pass


def build_summary_savers(model_dir):
    pass


class SummarySaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.savers = {}

    def add_saver(self, name, graph):
        self.savers[name] = tf.summary.FileWriter(os.path.join(self.model_dir, '{}_summaries'.format(name)), graph)

    def add_summary(self, name, summary, step):
        self.savers[name].add_summary(summary, step)


def build_trainer(logits, labels, global_step):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.identity(loss, name="loss")

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op, loss


inputs = build_inputs(X_train, y_train, 5)
steps = int(X_train.shape[0] / 5)


inputs_eval = build_inputs(X_test, y_test, 5)
steps_eval = int(X_test.shape[0] / 5)


def build_model(samples):
    out = samples

    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 50)

    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, 3)

    return out


samples = inputs['samples']
labels = inputs['labels']

with tf.variable_scope('model'):
    logits = build_model(samples)

with tf.variable_scope('model', reuse=True):
    eval_logits = build_model(inputs_eval['samples'])

with tf.variable_scope('optimizer'):
    global_step = tf.train.get_or_create_global_step()
    train_op, loss = build_trainer(logits, labels, global_step)

with tf.variable_scope("metrics"):
    print(tf.argmax(logits, axis=1))
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))
    loss, loss_update_op = tf.metrics.mean(loss)

with tf.variable_scope("metrics"):
    accurac, accuracy_update_o = tf.metrics.accuracy(inputs_eval['labels'], tf.argmax(eval_logits, axis=1))
    los, loss_update_o = tf.metrics.mean(loss)

update_metrics_op = tf.group(accuracy_update_op, loss_update_op)

metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
metrics_init_op = tf.variables_initializer(metric_variables)
data_iterator = inputs['iterator_init_op']
init = tf.global_variables_initializer()

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

with tf.variable_scope('savers'):
    last_saver = tf.train.Saver(name='last_saver')
    best_saver = tf.train.Saver(max_to_keep=1, name='best_saver')

begin_at_epoch = 0
model_dir = "logs"

best_eval_acc = 0.0
#metrics_values = {k: v[0] for k, v in metrics.items()}


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


global_step_val = 0


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)
    sess.run(init)

    for epoch in range(10):
        sess.run(data_iterator)
        sess.run(metrics_init_op)
        for step in range(steps):
            _, global_step_val, _ = sess.run([train_op, global_step, update_metrics_op])

        train_writer.add_summary(sess.run(summary_op), global_step_val)
        last_saver.save(sess, os.path.join(model_dir, 'last_weights', 'after-epoch'), epoch + 1)

        sess.run(data_iterator)
        sess.run(metrics_init_op)

        for _ in range(steps):
            sess.run(update_metrics_op)

        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        print("- Eval metrics: " + metrics_string)

        # global_step_val = sess.run(global_step)
        print(global_step_val)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            eval_writer.add_summary(summ, global_step_val)
        
        
        
        eval_acc = metrics_val['accuracy']

        if eval_acc >= best_eval_acc:
            best_eval_acc = eval_acc
            best_saver.save(sess, os.path.join(model_dir, 'best_weights', 'after-epoch'), epoch + 1)
