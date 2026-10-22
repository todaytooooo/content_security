"""WAV 声谱图与 MFCC 实验。"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import chirp, resample_poly


ROOT = Path(__file__).resolve().parent


def generate_demo(path):
    """三个各 4 秒的片段：440 Hz、880 Hz、300→3000 Hz 扫频。"""
    sr = 16000
    t = np.arange(4 * sr) / sr
    segments = [
        0.65 * np.sin(2 * np.pi * 440 * t),
        0.65 * np.sin(2 * np.pi * 880 * t),
        0.65 * chirp(t, f0=300, f1=3000, t1=4, method="linear"),
    ]
    # 每段首尾渐变 20 ms，减轻硬拼接造成的瞬态。
    fade = int(0.02 * sr)
    envelope = np.ones(t.size)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    samples = np.concatenate([s * envelope for s in segments])
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, np.round(samples * 32767).astype(np.int16))


def read_audio(path, target_sr):
    sr, samples = wavfile.read(path)
    dtype = samples.dtype
    if np.issubdtype(dtype, np.unsignedinteger): # unsigned int
        midpoint = (np.iinfo(dtype).max + 1) / 2 # 中点
        samples = (samples.astype(np.float64) - midpoint) / midpoint # 先减去中点，再除以中点，映射到约 [-1, 1]
    elif np.issubdtype(dtype, np.signedinteger):
        samples = samples.astype(np.float64) / (float(np.iinfo(dtype).max) + 1)
    elif np.issubdtype(dtype, np.floating):
        samples = samples.astype(np.float64)
    else:
        raise ValueError("不支持的 WAV 采样类型：" + str(dtype))
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    duration = len(samples) / sr
    if duration < 10:
        raise ValueError(f"音频时长为 {duration:.3f} 秒，实验要求不小于 10 秒。")
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-10:
        raise ValueError("音频包含非有限数值或为静音，请换用有效音频。")
    original_sr = sr
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        samples = resample_poly(samples, target_sr // divisor, sr // divisor)
    return samples, {
        "input_file": str(path.resolve()),
        "original_sample_rate": original_sr,
        "original_channels": channels,
        "original_dtype": str(dtype),
        "duration_seconds": duration,
        "sample_rate": target_sr,
        "num_samples": len(samples),
    }


def frame_audio(samples, frame_length, hop_length): # 分帧 帧长 frame_length（每帧的采样点数），帧移 hop_length（相邻帧起点间隔）
    # 不做中心补零；只在末尾补零，确保最后不足一帧的数据也被保留。
    count = 1 + max(0, math.ceil((len(samples) - frame_length) / hop_length))
    padded = np.pad(samples, (0, (count - 1) * hop_length + frame_length - len(samples)))
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None] #one（等价 np.newaxis）表示"在这里插入一个长度为 1 的新维度"
    return padded[indices]


def mel_filterbank(sr, n_fft, n_mels): # 梅尔滤波器组
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    boundaries = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)) # 边界频率，记录第 i 个滤波器的左/中/右 = boundaries[i], boundaries[i+1], boundaries[i+2]
    frequencies = np.fft.rfftfreq(n_fft, 1 / sr) # 一维数组
    filters = np.zeros((n_mels, len(frequencies))) 
    for i in range(n_mels):
        left, center, right = boundaries[i:i + 3]
        rising = (frequencies - left) / (center - left) #数组运算
        falling = (right - frequencies) / (right - center)
        filters[i] = np.maximum(0, np.minimum(rising, falling))
    if np.any(filters.sum(axis=1) == 0):
        raise ValueError("Mel 滤波器为空，请增大 FFT 点数或减少滤波器数量。")
    return filters, frequencies # 梅尔滤波器权重矩阵，FFT 频率轴


def extract_features(samples, sr=16000):
    frame_length, hop_length, n_fft = 400, 160, 512
    n_mels, n_mfcc, preemphasis = 40, 13, 0.97
    emphasized = np.append(samples[0], samples[1:] - preemphasis * samples[:-1]) #预加重，提升高频
    frames = frame_audio(emphasized, frame_length, hop_length) 
    windowed = frames * np.hamming(frame_length)
    # 所有后续特征使用同一份未做 dB 截断的功率谱。
    power = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) ** 2 / n_fft # FFT → 功率谱
    filters, frequencies = mel_filterbank(sr, n_fft, n_mels)
    mel_power = power @ filters.T
    log_mel = np.log(np.maximum(mel_power, 1e-12))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    reference = max(float(power.max()), 1e-12)
    # dB 图以该音频最大功率为 0 dB，显示时截取 [-80, 0] dB。
    spectrogram_db = 10 * np.log10(np.maximum(power, reference * 1e-8) / reference)
    times = (np.arange(len(frames)) * hop_length + frame_length / 2) / sr
    return {
        "power_spectrogram": power,
        "spectrogram_db": spectrogram_db,
        "mel_power": mel_power,
        "log_mel": log_mel,
        "mfcc": mfcc,
        "times_seconds": times,
        "frequencies_hz": frequencies,
        "mel_filters": filters,
    }, {
        "frame_length_samples": frame_length,
        "frame_length_ms": 1000 * frame_length / sr,
        "hop_length_samples": hop_length,
        "hop_length_ms": 1000 * hop_length / sr,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "n_mfcc": n_mfcc,
        "preemphasis": preemphasis,
        "window": "symmetric Hamming",
        "frame_padding": "right zero padding, no center padding",
        "mfcc_convention": "HTK Mel, unnormalized triangles, natural log floor 1e-12, orthonormal DCT-II, C0-C12",
        "array_layout": "frames x features",
    }


def save_plots(samples, sr, features, out):
    # 图中使用英文，避免系统未安装中文字体导致文字缺失。
    duration = len(samples) / sr
    extent = [0, (len(features["times_seconds"]) - 1) * 0.01 + 0.025, 0, sr / 2]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    stride = max(1, len(samples) // 60000)
    axes[0].plot(np.arange(len(samples))[::stride] / sr, samples[::stride], lw=0.5)
    axes[0].set(title="Input waveform", xlabel="Time (s)", ylabel="Amplitude", xlim=(0, duration))
    im = axes[1].imshow(features["spectrogram_db"].T, origin="lower", aspect="auto",
                        extent=extent, vmin=-80, vmax=0, cmap="magma")
    axes[1].set(title="Power spectrogram (pre-emphasized)", xlabel="Time (s)",
                ylabel="Frequency (Hz)", xlim=(0, duration))
    fig.colorbar(im, ax=axes[1], label="dB relative to peak")
    im = axes[2].imshow(features["mfcc"].T, origin="lower", aspect="auto",
                        extent=[extent[0], extent[1], -0.5, 12.5], cmap="coolwarm")
    axes[2].set(title="MFCC (C0-C12)", xlabel="Time (s)", ylabel="Coefficient index",
                yticks=range(13), xlim=(0, duration))
    fig.colorbar(im, ax=axes[2], label="Coefficient value")
    fig.savefig(out / "overview.png", dpi=180)
    plt.close(fig)

    for name, matrix, ylabel, limits, cmap in [
        ("spectrogram", features["spectrogram_db"].T, "Frequency (Hz)", [0, sr / 2], "magma"),
        ("mfcc", features["mfcc"].T, "MFCC coefficient index", [-0.5, 12.5], "coolwarm"),
        ("log_mel", features["log_mel"].T, "Mel filter index", [-0.5, 39.5], "magma"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        im = ax.imshow(matrix, origin="lower", aspect="auto",
                       extent=[extent[0], extent[1], *limits], cmap=cmap)
        ax.set(title=name.replace("_", " ").upper(), xlabel="Time (s)", ylabel=ylabel, xlim=(0, duration))
        fig.colorbar(im, ax=ax, label="dB relative to peak" if name == "spectrogram" else "Value")
        fig.savefig(out / (name + ".png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="不少于 10 秒的 WAV；省略时生成 12 秒示例")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="结果目录")
    args = parser.parse_args()
    audio_path = args.input
    if audio_path is None:
        audio_path = ROOT / "sample_audio" / "demo_12s.wav"
        if not audio_path.exists():
            generate_demo(audio_path)
    samples, metadata = read_audio(audio_path, target_sr=16000)
    features, parameters = extract_features(samples)
    if not all(np.isfinite(a).all() for a in features.values()):
        raise ValueError("特征包含非有限数值。")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", **features)
    np.savetxt(out / "mfcc.csv", features["mfcc"], delimiter=",",
               header=",".join(f"C{i}" for i in range(13)), comments="", fmt="%.8f")
    save_plots(samples, 16000, features, out)
    metadata.update(parameters)
    metadata["feature_shapes"] = {k: list(v.shape) for k, v in features.items()}
    metadata["mfcc_mean"] = features["mfcc"].mean(axis=0).tolist()
    metadata["mfcc_std"] = features["mfcc"].std(axis=0).tolist()
    metadata["all_features_finite"] = True
    (out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"结果已保存到：{out.resolve()}")


if __name__ == "__main__":
    main()
