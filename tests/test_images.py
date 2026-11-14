"""Image handling smoke tests; no model downloads or GPU required."""

import io
import unittest

import PIL
import torch
from packaging.version import Version
from PIL import Image, features

from feature_networks.clip.clip import _transform


class ImageCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Version(PIL.__version__) < Version("13.0.0"):
            raise RuntimeError("These tests require patched Pillow >=13.0.0")

    def test_clip_preprocessing_converts_and_resizes_rgba(self):
        image = Image.new("RGBA", (48, 32), (255, 0, 0, 128))
        tensor = _transform(16)(image)
        self.assertEqual(tuple(tensor.shape), (3, 16, 16))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())
        expected = (torch.tensor([1., 0., 0.]) - torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]
        )) / torch.tensor([0.26862954, 0.26130258, 0.27577711])
        torch.testing.assert_close(tensor[:, 0, 0], expected)

    def test_png_round_trip(self):
        image = Image.new("RGB", (16, 8), (17, 34, 51))
        output = io.BytesIO()
        image.save(output, format="PNG")
        output.seek(0)
        with Image.open(output) as restored:
            self.assertEqual(restored.size, image.size)
            self.assertEqual(restored.tobytes(), image.tobytes())

    def test_libtiff_rejects_inconsistent_scanline_geometry(self):
        # GHSA-xcvg-w56q-r89m: reject scanlines larger than the real row buffer.
        self.assertTrue(features.check("libtiff"), "libtiff coverage is required")
        image = Image.new("L", (64, 4), 17)
        for tags in ({256: 128}, {277: 3}):
            with self.subTest(tags=tags):
                output = io.BytesIO()
                with self.assertRaisesRegex(OSError, "encoder error -8"):
                    image.save(output, format="TIFF", compression="tiff_lzw", tiffinfo=tags)

    def test_libtiff_round_trip(self):
        self.assertTrue(features.check("libtiff"), "libtiff coverage is required")
        image = Image.new("RGB", (64, 4), (17, 34, 51))
        output = io.BytesIO()
        image.save(output, format="TIFF", compression="tiff_lzw", tiffinfo={317: 2})
        output.seek(0)
        with Image.open(output) as restored:
            self.assertEqual(restored.size, image.size)
            self.assertEqual(restored.tobytes(), image.tobytes())

    def test_resized_tiff_does_not_reuse_stale_exif_dimensions(self):
        self.assertTrue(features.check("libtiff"), "libtiff coverage is required")
        original = io.BytesIO()
        Image.new("RGB", (64, 4), (17, 34, 51)).save(
            original, format="TIFF", compression="tiff_lzw", tiffinfo={317: 2}
        )
        original.seek(0)
        with Image.open(original) as image:
            metadata = image.getexif()
            resized = image.resize((8, 2))
            output = io.BytesIO()
            with self.assertRaisesRegex(OSError, "encoder error -8"):
                resized.save(output, format="TIFF", compression="tiff_lzw", exif=metadata)


if __name__ == "__main__":
    unittest.main()
