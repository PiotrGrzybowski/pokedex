import unittest

from core.image import load_data, extract_extension, extract_label, extract_labels


class TestImage(unittest.TestCase):
    def test_extract_extension(self):
        self.assertEqual(extract_extension("/v.ar/fi.le.jpg"), ".jpg")
        self.assertEqual(extract_extension("/var/file.png"), ".png")
        self.assertEqual(extract_extension("a.png"), ".png")

    def test_extract_label(self):
        self.assertEqual(extract_label("/var/L1/img1.jpg"), "L1")

    def test_load_images(self):
        target_size = (224, 224)
        expected_images_shape = (6, 224, 224, 3)
        expected_labels_shape = (6, 3)

        images, labels = load_data('test_data/pokemon', target_size)

        self.assertEqual(expected_images_shape, images.shape)
        self.assertEqual(expected_labels_shape, labels.shape)


if __name__ == '__main__':
    unittest.main()
