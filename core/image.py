import cv2
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from imutils.paths import list_images
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def load_data(path, target_size):
    """
    Loads images and their labels into numpy arrays of given size.
    Pixels are normalized into [0, 1] range. Labels are saved in One Hot Encoding format.

    # Arguments
        path: Path to root directory with subdirectories which contain different class images.
        target_size: Tuple of ints `(img_height, img_width)`. Required because images can have different sizes.

    # Returns
        An images numpy array of shape (images_number, target_height, target_width, channels),
        an labels numpy array of shape (images_number, classes_number) in one hot encoding.
    """
    images, labels = [], []

    for filename in list_images(path):
        images.append(load_image(filename, target_size))
        labels.append(extract_label(filename))

    return normalize_pixels(images), encode_labels_to_one_hot(labels)


def normalize_pixels(images):
    """
    Normalizes image pixels into [0, 1] range from [0, 255].
 
    # Arguments
        An array of images.

    # Returns
        Normalized array of images in [0, 1] range.
    """
    return np.array(images, dtype="float") / 255.0


def encode_labels_to_one_hot(labels):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    # Arguments
        labels: List of 1d numpy array of labels. Labels must be hashable like integers and strings.

    # Returns

    """
    return LabelBinarizer().fit_transform(np.array(labels))


def load_image(filename, target_size):
    """
    Loads an image into numpy array of given size.

    # Arguments
        filename: Path to image file.
        target_size: Tuple of ints (target_height, target_width). Required because images can have different sizes.

    # Returns
        A numpy array of shape (target_height, target_width, channels).
    """
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image


def extract_label(filename):
    """
    Extract class label of the image from image path.

    # Arguments
        filename: Path to image file.

    # Returns
        Image class label.
    """
    return filename.split(os.path.sep)[-2]


def extract_labels(filenames):
    """
    Extract class labels of the images from image path list.

    # Arguments
        filenames: List of paths to image file.

    # Returns
        List of image labels.
    """
    return LabelEncoder().fit_transform([extract_label(filename) for filename in filenames])


def extract_extension(filename):
    return os.path.splitext(filename)[1]
