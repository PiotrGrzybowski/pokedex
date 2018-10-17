import argparse
import numpy as np
import os

from core.image import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Name of the data set.")
parser.add_argument("--input", required=True, help="Path to images root directory.")
parser.add_argument("--width", required=True, help="Target image width.")
parser.add_argument("--height", required=True, help="Target image height.")
parser.add_argument("--output", required=False, help="Path to save arrays. Equals input if None.")
args = vars(parser.parse_args())

name = args["name"]
input_path = args["input"]
target_size = (int(args["height"]), int(args["width"]))

if args["output"] is None:
    output_path = input_path
else:
    output_path = args["output"]

images, labels = load_data(input_path, target_size)
np.save(os.path.join(output_path, '{}_images_{}.npy'.format(name, target_size)), images)
np.save(os.path.join(output_path, '{}_labels.npy'.format(name)), labels)
