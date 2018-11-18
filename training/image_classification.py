import functools
import os
import numpy as np
import tensorflow as tf


def build_model_specification(inputs, model_fn, loss_fn, optimizer, reuse=False):
    specification = inputs

    with tf.variable_scope('model', reuse=reuse):
        logits = model_fn(specification['images'])

    loss = loss_fn(specification['labels'], logits)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss

    if optimizer:
        with tf.variable_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = get_train_op(optimizer, loss, tf.train.create_global_step())
                specification['train_op'] = train_op

    metrics = get_metrics(specification['labels'], logits, loss)

    specification['pred'] = tf.argmax(logits, axis=1)
    specification['conf'] = tf.confusion_matrix(specification['labels'], tf.argmax(logits, axis=1))
    specification['metrics'] = metrics
    specification['summary_op'] = get_summary_op(metrics)
    specification['metrics_init_op'] = get_metrics_init_op()
    specification['update_metrics_op'] = get_update_metrics_op(metrics)

    return specification


build_eval_model_specification = functools.partial(build_model_specification, optimizer=None, reuse=True)


def get_summary_op(metrics):
    return tf.summary.merge([tf.summary.scalar(name, metric[0]) for name, metric in metrics.items()])


def get_train_op(optimizer, loss, global_step):

    return optimizer.minimize(loss, global_step)


def get_update_metrics_op(metrics):
    return tf.group(*[op for _, op in metrics.values()])


def get_metrics(labels, logits, loss):
    with tf.variable_scope("metrics"):
        return {'accuracy': tf.metrics.accuracy(labels, tf.argmax(logits, axis=1)),
                'loss': tf.metrics.mean(loss)}


def get_metrics_init_op():
    return tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics"))


def build_model(inputs):
    out = inputs

    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 5)

    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, 3)

    return out


def build_inputs(features, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    irises, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    return {'samples': irises, 'labels': labels, 'iterator_init_op': iterator_init_op}


def get_savers(best_max_to_keep, last_max_to_keep):
    with tf.variable_scope('savers'):
        last_saver = tf.train.Saver(max_to_keep=last_max_to_keep, name='last_saver')
        best_saver = tf.train.Saver(max_to_keep=best_max_to_keep, name='best_saver')

    return best_saver, last_saver


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

            # eval_writer.add_summary(sess.run(eval_spec['summary_op']), global_step_val)

            eval_metrics = eval_spec['metrics']
            metrics_values = {k: v[0] for k, v in eval_metrics.items()}
            metrics_val = sess.run(metrics_values)

            # Add summaries manually to writer at global_step_val
            global_step_val = sess.run(global_step)
            for tag, val in metrics_val.items():
                summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                eval_writer.add_summary(summ, global_step_val)

            eval_accuracy = sess.run(eval_spec['metrics']['accuracy'][0])

            # if epoch > 0 and epoch % 10 == 0:
            print("Epoch {}, accuracy = {}".format(epoch, eval_accuracy))
            if eval_accuracy >= best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                best_saver.save(sess, os.path.join(model_dir, 'best_weights', 'after-epoch'), epoch + 1)

        matrix = np.zeros((10, 10), dtype=np.int)

        sess.run(eval_spec['iterator_init_op'])
        sess.run(eval_spec['metrics_init_op'])

        for _ in range(eval_steps_per_epoch):
            matrix += sess.run(eval_spec['conf'])

        print(matrix)
        print(np.sum(matrix))
        print(np.trace(matrix))