import tensorflow as tf

from abc import ABC, abstractmethod
from google.protobuf import text_format
from proto import label_index_map_pb2


class LabelMap(ABC):
    def __init__(self, path):
        super().__init__()
        self.index_to_label_dict = None
        self.label_to_index_dict = None
        self.load_label_map(path)

    @abstractmethod
    def load_label_map(self, path):
        pass

    def get_index_from_label(self, label):
        return self.label_to_index_dict[label]

    def get_label_from_index(self, index):
        return self.index_to_label_dict[index]


class ProtoLabelMap(LabelMap):
    def load_label_map(self, path):
        """
        Loads label map proto.

        Arguments:
          path: path to StringIntLabelMap proto text file.

        Returns:
          a LabelIndexMapProto
        """
        with tf.gfile.GFile(path, 'r') as map_file:
            label_map_string = map_file.read()
            label_map = label_index_map_pb2.LabelIndexMap()
            text_format.Merge(label_map_string, label_map)

            self.label_to_index_dict = {item.name: item.id for item in label_map.item}
            self.index_to_label_dict = {item.id: item.name for item in label_map.item}
