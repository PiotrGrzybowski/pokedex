import tensorflow as tf
import os
import numpy as np

def evaluate(eval_spec, model_dir):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(tf.global_variables_initializer())

        # Reload weights from the weights subdirectory
        save_path = model_dir
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        sess.run(eval_spec['iterator_init_op'])
        sess.run(eval_spec['metrics_init_op'])

        matrix = tf.confusion_matrix(sess.run(eval_spec['labels']), sess.run(eval_spec['logits']))
        matrix = sess.run(matrix)
        print(np.sum(matrix))
        print(np.trace(matrix))
            # num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        # metrics = evaluate_sess(sess, model_spec, num_steps)
        # metrics_name = '_'.join(restore_from.split('/'))
        # save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        # save_dict_to_json(metrics, save_path)