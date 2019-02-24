import gzip
import logging
import os
import shutil
import tensorflow as tf
from six.moves import urllib

MNIST_MIRROR = 'http://yann.lecun.com/exdb/mnist/'
FI = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MnistLoader")


def download(download_links, zipped_paths):
    for link, zipped_path in zip(download_links, zipped_paths):
        logger.info('Downloading {} to {}'.format(link, zipped_path))
        urllib.request.urlretrieve(link, zipped_path)


def extract(zipped_paths, extracted_paths):
    for zipped_path, extracted_path in zip(zipped_paths, extracted_paths):
        with gzip.open(zipped_path, 'rb') as file_in, tf.gfile.Open(extracted_path, 'wb') as file_out:
            logger.info('Extracting {}'.format(zipped_path))
            shutil.copyfileobj(file_in, file_out)


def remove(zipped_paths):
    for zipped_path in zipped_paths:
        logger.info('Removing {}'.format(zipped_path))
        os.remove(zipped_path)


def decode_image_mlp(image):
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0


def decode_image_cnn(image):
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [28, 28, 1])
    return image / 255.0


def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [])
    label = tf.one_hot(label, 10)
    return label


def get_dataset(images_file, labels_file, decode_image, cores):
    images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16)
    images = images.map(decode_image, num_parallel_calls=cores)

    labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8)
    labels = labels.map(decode_label, num_parallel_calls=cores)
    return tf.data.Dataset.zip((images, labels))


def get_inputs(images_file, labels_file, batch_size, buffer_size, prefetch, cores, decode_image):
    dataset = get_dataset(images_file, labels_file, decode_image, cores)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    return images, labels, iterator_init_op


def get_inputs_mlp(images_file, labels_file, batch_size, prefetch, cores, buffer_size):
    return get_inputs(images_file, labels_file, batch_size, prefetch, cores, buffer_size, decode_image_mlp)


def get_inputs_cnn(images_file, labels_file, batch_size, prefetch, cores, buffer_size):
    return get_inputs(images_file, labels_file, batch_size, prefetch, cores, buffer_size, decode_image_cnn)


def download_missing_files(directory, files_to_download):
    download_links = [os.path.join(MNIST_MIRROR, filename + '.gz') for filename in files_to_download]
    zipped_paths = [os.path.join(directory, filename + '.gz') for filename in files_to_download]
    extracted_paths = [os.path.join(directory, filename) for filename in files_to_download]
    download(download_links, zipped_paths)
    extract(zipped_paths, extracted_paths)
    remove(zipped_paths)


def paget(directory, build_inputs, batch_size, prefetch, cores):
    files = [os.path.join(directory, filename) for filename in FI]
    files_to_download = [filename for filename in FI if not tf.gfile.Exists(os.path.join(directory, filename))]

    if files_to_download:
        download_missing_files(directory, files_to_download)

    train_images, train_labels, train_iterator_init = build_inputs(files[0], files[1], batch_size, prefetch, cores, 60000)
    test_images, test_labels, test_iterator_init = build_inputs(files[2], files[3], batch_size, prefetch, cores, 10000)

    return train_images, train_labels, train_iterator_init, test_images, test_labels, test_iterator_init


def paget_mlp(directory, batch_size, prefetch, cores):
    return paget(directory, get_inputs_mlp, batch_size, prefetch, cores)


def paget_cnn(directory, batch_size, prefetch, cores):
    return paget(directory, get_inputs_cnn, batch_size, prefetch, cores)


if __name__ == '__main__':
    destination = '/Users/Piotr/Workspace/DataScience/pokedex/datasets'

    # images, labels, iterator_init_op = get_inputs_mlp(extracted_paths[0], extracted_paths[1], 10, 60000)
    # with tf.Session() as sess:
    #     sess.run(iterator_init_op)
    #     im, la = sess.run([images, labels])
    #     print(im.shape)

    x_train, y_train, init_train, x_test, y_test, init_test = paget_cnn(destination, 32, prefetch=2, cores=4)

    with tf.Session() as sess:
        sess.run([init_train, init_test])

        print(sess.run(x_train).shape)
        print(sess.run(y_train).shape)
