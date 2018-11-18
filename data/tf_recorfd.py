import numpy as np
import tensorflow as tf

from core.image import load_image, extract_label
from google.protobuf import text_format
from proto.record_config_pb2 import RecordConfig


def load_data_tf_record_config(path):
    """
    Loads tf record config proto.

    Arguments:
      path: path to DataTfRecordConfig proto text file.
    """
    with tf.gfile.GFile(path, 'r') as config_file:
        data_config = RecordConfig()
        text_format.Merge(config_file.read(), data_config)

        data_path = data_config.data_path
        output_path = data_config.output_path
        target_size = (data_config.target_size[0], data_config.target_size[1])

        return data_path, output_path, target_size


def build_record_example(image, label):
    example = tf.train.Example(features=build_features(image, label))
    return example.SerializeToString()


def build_features(image, label):
    return tf.train.Features(feature={'label': int64_feature(label), 'image': bytes_feature(image)})


def bytes_feature(image):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_image_to_string(filename, target_size):
    return tf.image.encode_jpeg(load_image(filename, target_size))


def generate_image_label_pairs(image_paths, label_map, target_size):
    for filename in image_paths:
        label = label_map.get_index_from_label(extract_label(filename))
        image = load_image_to_string(filename, target_size)

        yield image, label


def parse_cifar_record(record):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=-1)}

    parsed_features = tf.parse_single_example(record, features)
    image = tf.decode_raw(parsed_features["image"], out_type=tf.int8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image.set_shape((3072, ))
    image = tf.manip.reshape(image, [32, 32, 3])
    print(image)
    label = parsed_features["label"]

    return image, label


def preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.5)
    image = tf.clip_by_value(image, 0.0, 1.0)


    return image, label

def build_inputs_from_cifar_tf_record_data(path, batch_size, cores, aug=False):
    dataset = tf.data.TFRecordDataset(path, num_parallel_reads=cores)
    dataset = dataset.map(parse_proto_image, num_parallel_calls=cores)
    if aug:
        dataset = dataset.map(preprocess, num_parallel_calls=cores)
    dataset = dataset.shuffle(50000)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer
    # images.set_shape((batch_size, 3072))
    labels.set_shape((batch_size,))

    return {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}


def parse_proto_image(record):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=-1)}

    parsed_features = tf.parse_single_example(record, features)

    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image.set_shape([3 * 32 * 32])
    image = tf.cast(tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]), tf.float32)
    image = image / 255.0

    label = tf.cast(parsed_features["label"], tf.int32)

    return image, label


def build_inputs_from_tf_record_data(path, batch_size, cores):
    dataset = tf.data.TFRecordDataset(path, num_parallel_reads=cores)
    dataset = dataset.map(parse_proto_image, num_parallel_calls=cores)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    return {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
