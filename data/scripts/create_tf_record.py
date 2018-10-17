import argparse
import os
import tensorflow as tf

from data.label_map import ProtoLabelMap
from data.tf_recorfd import load_data_tf_record_config, generate_image_label_pairs, build_record_example
from utils.file import split_test_train

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, help="Config path.")
args = vars(parser.parse_args())

data_path, output_path, target_size = load_data_tf_record_config(args["config_path"])
label_map = ProtoLabelMap(os.path.join(data_path, "label_map.pbtxt"))


train_paths, test_paths = split_test_train(data_path, 0.8)


def create_tf_record_data_set(sess, image_paths, output_path):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for image, label in generate_image_label_pairs(image_paths, label_map, target_size):
            image = sess.run(image)
            example = build_record_example(image, label)
            writer.write(example)


with tf.Session() as sess:
    create_tf_record_data_set(sess, train_paths, output_path + "-train.record")
    create_tf_record_data_set(sess, test_paths, output_path + "-test.record")

