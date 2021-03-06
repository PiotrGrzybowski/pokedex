import functools

import tensorflow as tf


def build_model_specification(inputs, model_fn, loss_fn, optimizer, reuse=False):
    specification = inputs

    with tf.variable_scope('model', reuse=reuse):
        logits = model_fn(specification['samples'])

    loss = loss_fn(specification['labels'], logits)

    if optimizer:
        with tf.variable_scope('optimizer'):
            specification['train_op'] = get_train_op(optimizer, loss, tf.train.create_global_step())

    metrics = get_metrics(specification['labels'], logits, loss)

    specification['metrics'] = metrics
    specification['summary_op'] = get_summary_op(metrics)
    specification['metrics_init_op'] = get_metrics_init_op()
    specification['update_metrics_op'] = get_update_metrics_op(metrics)

    return specification


eval_specification = functools.partial(build_model_specification, optimizer=None, reuse=True)


def get_summary_op(metrics):
    return tf.summary.merge([tf.summary.scalar(name, metric[0]) for name, metric in metrics.items()])


def get_train_op(optimizer, loss, global_step):
    return optimizer.minimize(loss, global_step)


def get_update_metrics_op(metrics):
    return tf.group(*[op for _, op in metrics.values()])


def get_metrics(labels, logits, loss):
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels, tf.argmax(logits, axis=1)),
            'recall': tf.metrics.recall(labels, tf.argmax(logits, axis=1)),
            'precision': tf.metrics.precision(labels, tf.argmax(logits, axis=1)),
            'loss': tf.metrics.mean(loss)
        }
    return metrics


def get_metrics_init_op():
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    return tf.variables_initializer(metric_variables)


def build_model(samples):
    out = samples

    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 5)

    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, 3)

    return out


def build_inputs(features, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    irises, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    return {'samples': irises, 'labels': labels, 'iterator_init_op': iterator_init_op}


def get_savers(best_max_to_keep, last_max_to_keep):
    with tf.variable_scope('savers'):
        last_saver = tf.train.Saver(max_to_keep=last_max_to_keep, name='last_saver')
        best_saver = tf.train.Saver(max_to_keep=best_max_to_keep, name='best_saver')

    return best_saver, last_saver


import os


def get_summary_writers(sess, model_dir):
    train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

    return train_writer, eval_writer


def train(train_spec, eval_spec, model_dir, epochs, train_steps_per_epoch, eval_steps_per_epoch):
    global_step = tf.train.get_global_step()
    global_step_val = None
    best_eval_accuracy = 0.0
    best_saver, last_saver = get_savers(1, 5)
    begin_at_epoch = 0

    with tf.Session() as sess:
        train_writer, eval_writer = get_summary_writers(sess, model_dir)
        sess.run(tf.global_variables_initializer())

        for epoch in range(begin_at_epoch, begin_at_epoch + epochs):
            sess.run(train_spec['iterator_init_op'])
            sess.run(train_spec['metrics_init_op'])

            for step in range(train_steps_per_epoch):
                _, global_step_val, _ = sess.run([train_spec['train_op'], global_step, train_spec['update_metrics_op']])

            train_writer.add_summary(sess.run(train_spec['summary_op']), global_step_val)
            last_saver.save(sess, os.path.join(model_dir, 'last_weights', 'after-epoch'), epoch + 1)

            sess.run(eval_spec['iterator_init_op'])
            sess.run(eval_spec['metrics_init_op'])

            for _ in range(eval_steps_per_epoch):
                sess.run(eval_spec['update_metrics_op'])

            eval_writer.add_summary(sess.run(eval_spec['summary_op']), global_step_val)
            eval_accuracy = sess.run(eval_spec['metrics']['accuracy'][0])

            if eval_accuracy >= best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                best_saver.save(sess, os.path.join(model_dir, 'best_weights', 'after-epoch'), epoch + 1)
