import json
import os


def split_test_train(data_path, split_factor):
    train_paths = []
    test_paths = []
    categories = [os.path.join(data_path, directory) for directory in os.listdir(data_path) if
                  os.path.isdir(os.path.join(data_path, directory))]

    for directory in categories:
        paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        train_paths.extend(paths[:int(len(paths) * split_factor)])
        test_paths.extend(paths[int(len(paths) * split_factor):])

    return train_paths, test_paths


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
