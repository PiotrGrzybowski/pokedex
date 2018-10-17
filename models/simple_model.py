import tensorflow as tf


def build_model(inputs):
    images = inputs['images']

    out = images
    num_channels = 16
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]

    for i, filters in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i + 1)):
            out = tf.layers.conv2d(out, filters=filters, kernel_size=3, padding='same')
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)

    out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 8)
        out = tf.nn.relu(out)

    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, 5)
    return logits


def model(inputs):
    out = inputs["images"]

    with tf.variable_scope('block_1'):
        out = tf.layers.conv2d(out, filters=16, kernel_size=3, padding='same')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)

    with tf.variable_scope('block_2'):
        out = tf.layers.conv2d(out, filters=32, kernel_size=3, padding='same')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)

    with tf.variable_scope('block_3'):
        out = tf.layers.conv2d(out, filters=64, kernel_size=3, padding='same')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)

    out = tf.reshape(out, [-1, 8 * 8 * 64])

    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 64)
        out = tf.nn.relu(out)

    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, 5)
        out = tf.nn.relu(out)

    return out


def model_train_specification(inputs):
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    with tf.variable_scope('model'):
        logits = model(inputs)
        predictions = tf.argmax(logits, 1)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    print(loss)
    print(global_step)
    print(train_op)

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    mask = tf.not_equal(labels, predictions)
    for label in range(0, 5):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['train_op'] = train_op
    print(model_spec['variable_init_op'])

    return model_spec


def eval_model_specification(inputs):
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    with tf.variable_scope('model', reuse=True):
        logits = model(inputs)
        predictions = tf.argmax(logits, 1, name='predictions')

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    mask = tf.not_equal(labels, predictions)
    for label in range(0, 5):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    return model_spec