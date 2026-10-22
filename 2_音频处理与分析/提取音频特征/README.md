# 音频 MFCC 与声谱图特征提取实验

实验任务：提取任意一段不少于 10 秒的 WAV 音频的 MFCC 特征和声谱图特征，理解两类特征的原理。

本实验已使用程序生成的 **12 秒、16 kHz、16 bit、单声道 WAV** 完成一次运行。音频是合成测试信号，不是真人语音或实地录音，无需下载外部数据。它的频率变化已知，适合验证特征是否正确。

## 运行方法

在 `2_音频处理与分析/提取音频特征` 目录打开终端：

```powershell
python -m pip install -r requirements.txt
python audio_features.py
```

也可以从项目根目录运行：

```powershell
python "2_音频处理与分析/提取音频特征/audio_features.py"
```

默认音频保存在脚本所在目录的 `sample_audio/demo_12s.wav`，结果保存在 `results/`，不受终端当前目录影响。已有示例 WAV 会直接读取；删除该示例后再运行可重新生成。

使用自己的 WAV：

```powershell
python audio_features.py --input "D:\音频\自己的录音.wav" --output "results_my_audio"
```

输入必须不少于 10 秒，程序不会通过重复短音频凑时长。多声道先取通道均值，随后按需重采样为 16 kHz；程序支持整数 PCM 和浮点 WAV，具体格式以 SciPy 的 WAV 读取能力为准。若立体声两个通道反相，取均值可能相消，应先选择有效声道。采样率统一后，8 kHz 以上的信息不会保留。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `audio_features.py` | 生成音频、预处理、分帧、FFT、Mel 滤波、DCT、绘图与保存 |
| `requirements.txt` | NumPy、SciPy、Matplotlib 依赖 |
| `实验原理与结果.md` | 原理、公式、参数、实验过程和结果解释 |
| `sample_audio/demo_12s.wav` | 可直接播放的 12 秒实验输入 |
| `results/overview.png` | 波形、声谱图、MFCC 三联图 |
| `results/spectrogram.png` | 单独的功率声谱图 |
| `results/mfcc.png` | 单独的 MFCC 热力图 |
| `results/log_mel.png` | MFCC 的中间结果：对数 Mel 能量图 |
| `results/features.npz` | 全部数值特征，含时间、频率坐标与 Mel 滤波器 |
| `results/mfcc.csv` | 1199 行、13 列 MFCC，首行为 C0～C12 列名 |
| `results/summary.json` | 输入信息、提取参数、矩阵维度及统计量 |

读取特征：

```python
import numpy as np

with np.load("results/features.npz") as features:
    mfcc = features["mfcc"]                      # [时间帧, 倒谱系数]
    power = features["power_spectrogram"]        # [时间帧, 频率点]
    db = features["spectrogram_db"]              # 相对峰值 dB，截断到 -80 dB
    times = features["times_seconds"]            # 每帧中心时刻，单位 s
    frequencies = features["frequencies_hz"]     # 频率坐标，单位 Hz
    print(mfcc.shape, power.shape)               # (1199, 13) (1199, 257)
```

数值矩阵按“帧×特征”保存，绘图时转置为“纵轴特征、横轴时间”。图片使用英文坐标避免中文字体缺失，中文解释见实验说明。已有 Word 模板保留在上一级目录 `../实验报告模板 (1).docx`，可将说明中的内容及结果图片填入模板。

![波形、声谱图与 MFCC](results/overview.png)
