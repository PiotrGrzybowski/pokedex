import tensorflow as tf6 0 1 1 8

from models.training import train_and_evaluate, train


def preprocess_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


if __name__ == '__main__':
    train_path = '/DATA/piotr/pokedex/data/FivePokemon/data_64x64-train.record'
    test_path = '/DATA/piotr/pokedex/data/FivePokemon/data_64x64-test.record'

    train_inputs = build_inputs_from_tf_record_data(train_path, 32, 8)
    test_inputs = build_inputs_from_tf_record_data(test_path, 32, 8)

    train(train_inputs, test_inputs)

