"""Small, offline tests. No dataset download or training run required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
import soundfile as sf

import audio_classification as audio


class AudioClassificationTests(unittest.TestCase):
    def test_smoke(self):
        audio.smoke_test()

    def test_pcm16_stereo_resampling_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stereo.wav"
            samples = np.full((8000, 2), 16384, dtype=np.int16)
            wavfile.write(path, 8000, samples)
            loaded = audio.load_audio(path)
            self.assertEqual(loaded.shape, (64000,))
            self.assertAlmostEqual(float(loaded[100:15000].mean()), 0.5, places=3)
            self.assertTrue(np.all(loaded[16000:] == 0))

    def test_unsigned_pcm(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "silence.wav"
            wavfile.write(path, 16000, np.full(100, 128, dtype=np.uint8))
            self.assertTrue(np.all(audio.load_audio(path) == 0))

    def test_truncates_long_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "long.wav"
            wavfile.write(path, 16000, np.ones(80000, dtype=np.float32))
            loaded = audio.load_audio(path)
            self.assertEqual(loaded.shape, (64000,))
            self.assertTrue(np.all(loaded == 1))

    def test_rejects_nonfinite_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nan.wav"
            wavfile.write(path, 16000, np.array([np.nan], dtype=np.float32))
            with self.assertRaises(ValueError):
                audio.load_audio(path)

    def test_rejects_same_validation_test(self):
        with self.assertRaises(ValueError):
            audio.split_masks(np.arange(1, 11), 10, 10)

    def test_adpcm_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adpcm.wav"
            wave = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.5
            sf.write(path, wave, 16000, subtype="IMA_ADPCM")
            loaded = audio.load_audio(path)
            self.assertEqual(loaded.shape, (64000,))
            self.assertTrue(np.isfinite(loaded).all())
            self.assertGreater(float(np.std(loaded[:15000])), 0.2)

    def test_saved_model_roundtrip(self):
        torch.manual_seed(42)
        model = audio.LeNet5().eval()
        features = torch.randn(2, 1, 64, 128)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = audio.LeNet5().eval()
            restored.load_state_dict(audio.load_checkpoint(path)["state_dict"])
            with torch.no_grad():
                torch.testing.assert_close(model(features), restored(features))


if __name__ == "__main__":
    unittest.main()
