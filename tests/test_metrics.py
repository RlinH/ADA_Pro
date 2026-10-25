import unittest

import numpy as np

from dgseg.metrics import Confusion
from dgseg.labels import map_mask


class MetricsTests(unittest.TestCase):
    def test_global_confusion_ignore_and_absent(self):
        metric = Confusion(3)
        metric.update(np.array([[0, 1], [1, 2]]), np.array([[0, 1], [0, 255]]))
        metric.update(np.array([[1, 1]]), np.array([[1, 1]]))
        report = metric.summary()
        self.assertEqual(report["confusion_matrix"], [[1, 1, 0], [0, 3, 0], [0, 0, 0]])
        self.assertEqual(report["valid_pixels"], 5)
        self.assertEqual(report["ignored_pixels"], 1)
        self.assertEqual(report["included_class_ids"], [0, 1])
        self.assertAlmostEqual(report["miou"], (0.5 + 0.75) / 2)
        self.assertAlmostEqual(report["pixel_accuracy"], 4 / 5)

    def test_loveda_seven_classes_and_unknown(self):
        np.testing.assert_array_equal(map_mask(np.array([[0, 1, 7]], dtype=np.uint8), "loveda"), [[255, 0, 6]])
        with self.assertRaises(ValueError):
            map_mask(np.array([[8]], dtype=np.uint8), "loveda")

    def test_mapillary_indexed_labels(self):
        np.testing.assert_array_equal(map_mask(np.array([[13, 55, 65]], dtype=np.uint8), "mapillary_v1.2"), [[0, 13, 255]])
        with self.assertRaises(ValueError):
            map_mask(np.array([[66]], dtype=np.uint8), "mapillary_v1.2")


if __name__ == "__main__":
    unittest.main()
