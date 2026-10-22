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

    def test_gap_head_training_and_checkpoint(self):
        torch.manual_seed(42)
        model = audio.LeNet5(head="gap")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4310)
        inputs = torch.randn(2, 1, 64, 128)
        before = model.features[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([1, 9]))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.features[0].weight))
        model.eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gap.pt"
            torch.save(dict(state_dict=model.state_dict(), config=dict(head="gap")), path)
            restored = audio.model_from_checkpoint(audio.load_checkpoint(path)).eval()
            with torch.no_grad():
                self.assertEqual(tuple(restored(inputs).shape), (2, 10))
                torch.testing.assert_close(model(inputs), restored(inputs))
                torch.testing.assert_close(model(inputs), model(inputs))

    def test_legacy_checkpoint_model_selection(self):
        model = audio.LeNet5()
        restored = audio.model_from_checkpoint(dict(state_dict=model.state_dict(), config={}))
        self.assertEqual(sum(p.numel() for p in restored.parameters()), 737546)

    def test_time_pooling_preserves_frequency_and_checkpoint(self):
        model = audio.LeNet5(head="time")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11990)
        pool = model.classifier[2]
        # A frequency ramp must survive pooling, while time is averaged away.
        frequency = torch.arange(13.0).view(1, 1, 13, 1)
        time = torch.arange(29.0).view(1, 1, 1, 29)
        torch.testing.assert_close(pool(frequency + time), frequency + 14)
        inputs = torch.randn(2, 1, 64, 128)
        before = model.features[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([1, 9]))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.features[0].weight))
        model.eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "time.pt"
            torch.save(dict(state_dict=model.state_dict(), config=dict(head="time")), path)
            restored = audio.model_from_checkpoint(audio.load_checkpoint(path)).eval()
            with torch.no_grad():
                self.assertEqual(tuple(restored(inputs).shape), (2, 10))
                torch.testing.assert_close(model(inputs), restored(inputs))


if __name__ == "__main__":
    unittest.main()
