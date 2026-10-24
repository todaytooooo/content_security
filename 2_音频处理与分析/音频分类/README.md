# UrbanSound8K：MFCC + 2D CNN 音频分类

本目录仅保留 MFCC + 轻量 2D CNN，以及当前 fold10 上表现最好的 seed 7 完整实验产物。

## 文件结构

- `audio_classification.py`：音频读取、MFCC 提取、训练、评估、预测。
- `test_audio_classification.py`：数据处理与模型测试。
- `requirements.txt`：运行依赖。
- `UrbanSound8K/`、`UrbanSound8K.rar`：原始数据及压缩包。
- `cache/mfcc_v1.npz`：MFCC 特征缓存。
- `outputs/mfcc_cnn_seed7/test_fold10/`：唯一保留的完整实验结果。
- [实验报告.md](实验报告.md)：模型、数据协议、结果及局限。

## 使用

在本目录执行：

```powershell
$AudioPython = "D:\Tools\Anaconda\envs\content-security\python.exe"
& $AudioPython -m unittest -v test_audio_classification.py
& $AudioPython audio_classification.py evaluate --checkpoint outputs/mfcc_cnn_seed7/test_fold10/best.pt
& $AudioPython audio_classification.py predict --checkpoint outputs/mfcc_cnn_seed7/test_fold10/best.pt --audio "实际音频路径.wav"
# 默认 MFCC + CNN，seed 7；新目录用于避免覆盖已保存的权重
& $AudioPython -u audio_classification.py train --output outputs/mfcc_cnn_repeat
```

其他 Python 环境需安装 requirements.txt 中的依赖。训练自动使用可用的 CUDA，评估与预测使用 CPU。默认数据和缓存路径以脚本所在目录为基准。

## 结果文件

Accuracy **63.20%**（529/837），Macro-F1 **63.27%**。训练 60 轮，以最低验证损失选中第 58 轮。

- `best.pt`：权重、训练集标准化统计量和配置。
- `config.json`：参数、数据数量、种子、设备及软件版本。
- `split_manifest.csv`：逐样本训练/验证/测试/剔除归属。
- `history.json`、`training_curves.png`：训练与验证历史。
- `metrics.json`、`test_predictions.csv`、`confusion_matrix.png`：测试结果。
- `mfcc_example.png`：输入特征示例。

seed 7 是已观察的三个运行中测试指标最高的一次，这个单次结果不能当成无偏的总体平均性能。详见实验报告。
