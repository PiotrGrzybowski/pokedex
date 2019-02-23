import gzip
import logging
import os
import shutil
import tempfile
import tensorflow as tf
from six.moves import urllib


MNIST_MIRROR = 'http://yann.lecun.com/exdb/mnist/'
FILES = {'train_images': 'train-images-idx3-ubyte.gz',
         'train_labels': 'train-labels-idx1-ubyte.gz',
         'test_images': 't10k-images-idx3-ubyte.gz',
         'test_labels': 't10k-labels-idx1-ubyte.gz'}


def downd(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath

def download(directory)


if __name__ == '__main__':

    dest = '/home/piotr/Workspace/Projects/pokedex/datasets'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not tf.gfile.Exists(dest):
        tf.gfile.MkDir(dest)
    else:
        for key, filename in FILES.items():
            if not tf.gfile.Exists(os.path.join(dest, filename)):
                url = os.path.join(MNIST_MIRROR, filename)
                logging.info('Downloading {} to {}'.format(url, os.path.join(dest, filename)))
                urllib.request.urlretrieve(url, os.path.join(dest, filename))

