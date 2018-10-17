import logging
import os

import tensorflow as tf
from tqdm import trange

from models.evaluation import evaluate_sess
from models.simple_model import model
from utils.file import save_dict_to_json


def train(train_inputs, test_inputs):
    train_images = train_inputs["images"]
    train_labels = tf.cast(train_inputs["labels"], tf.int64)
    train_iterator_init_op = train_inputs["iterator_init_op"]

    with tf.variable_scope('model'):
        logits = model(train_inputs)
    global_step = tf.train.get_or_create_global_step()
    loss, train_op = build_optimizer(logits, train_labels, global_step)

    # print(loss)
    # print(tf.global_variables_initializer())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for jk in range(30):
            sess.run(train_iterator_init_op)
            for i in range(50):
                _, loss_val = sess.run([train_op, loss])
                print(loss_val)


def build_optimizer(logits, labels, global_step):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)
    return loss, train_op


def train_sess(sess, model_spec, num_steps, writer):
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    t = trange(50)

    for i in t:
        _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss, summary_op, global_step])
        writer.add_summary(summ, global_step_val)

        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    # print(metrics.items())
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(train_model_spec, eval_model_spec):
    last_saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=1)
    begin_at_epoch = 0
    model_dir = "/DATA/piotr/pokedex/learning/FivePokemon"
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)

        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch, 20):
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + 20))
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (1877 + 32 - 1) // 32
            train_sess(sess, train_model_spec, num_steps, train_writer)

            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)

            num_steps = (472 + 32 - 1) // 32
            metrics = evaluate_sess(sess, eval_model_spec, num_steps, eval_writer)

            # If best_eval, best_save_path
            eval_acc = metrics['accuracy']
            if eval_acc >= best_eval_acc:
                # Store new best accuracy
                best_eval_acc = eval_acc
                # Save weights
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
                save_dict_to_json(metrics, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)


def create_summary_writer(directory, graph):
    return tf.summary.FileWriter(directory, graph)


def train_epoch(sess, steps, writer):
    for step in range(steps):
        _, _, loss_val, summary, global_step_val = sess.run([train_op, update_metrics, loss, summary_op, global_step])

        writer.add_summary(summary, global_step_val)
    pass


def evaluate_epoch():
    pass


def accuracy(out, labels):
    return tf.metrics.accuracy(labels=labels, predictions=tf.argmax(out, 1))


def metrics(labels, predictions, loss):
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    return metrics, update_metrics_op, metrics_init_op






    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     best_eval_acc = 0.0
    #
    #     for epoch in range(begin_at_epoch, begin_at_epoch + epochs):
    #         train_epoch()
    #         evaluate_epoch()
    #
    #         _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss, summary_op, global_step])
