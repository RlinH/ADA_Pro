from pathlib import Path
import tempfile
import unittest
import zipfile

from dgseg.data import source_samples


class SourceIsolationTests(unittest.TestCase):
    def test_loveda_source_catalog_needs_no_target_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "downloads/loveda/Train.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as zf:
                for index in range(10):
                    zf.writestr(f"Train/Urban/images_png/{index}.png", b"placeholder")
                    zf.writestr(f"Train/Urban/masks_png/{index}.png", b"placeholder")
            groups = source_samples(root, "rs7_u2r", seed=42)
            self.assertEqual(set(groups), {"source_train", "source_val"})
            self.assertEqual((len(groups["source_train"]), len(groups["source_val"])), (9, 1))
            self.assertTrue(all(sample["image_archive"].endswith("Train.zip") for rows in groups.values() for sample in rows))


if __name__ == "__main__":
    unittest.main()
