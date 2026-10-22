# UrbanSound8K 音频分类：LeNet-5 复现与音频适配

本实验从零训练 LeNet-5 的音频适配版本，把环境声音转换成二维 Log-Mel 声谱图，再进行 10 分类。不使用预训练权重。实际运行结果、实验原理和局限见 [实验报告.md](实验报告.md)。

## 1. 文件说明

- `audio_classification.py`：数据校验、特征提取、训练、评估、单音频预测和自检。
- `test_audio_classification.py`：合成音频单元测试；不把合成数据当作真实实验结果。
- `requirements.txt`：运行依赖。
- `cache/logmel_v1.npz`：首次运行产生的特征缓存，可重新生成。
- `outputs/lenet5/test_fold10/`：真实训练结果，包括模型和图表。
- `UrbanSound8K/`：现有解压数据；程序自动查找嵌套目录中的唯一元数据文件。

程序默认路径均以脚本所在目录为基准，从其他工作目录调用也可以。缓存和大型数据、模型不加入 Git。

## 2. 运行环境

推荐 Python 3.10/3.11，支持 CPU 和 CUDA。当前机器已有可用环境：

```powershell
cd "D:\2026-2027大三上\content_security\2_音频处理与分析\音频分类"
$AudioPython = "D:\Tools\Anaconda\envs\content-security\python.exe"
& $AudioPython audio_classification.py smoke-test
& $AudioPython -m unittest -v test_audio_classification.py
```

这里不使用 Anaconda base 环境：检查时其 PyTorch 无法加载 DLL。没有上述环境时，可在独立虚拟环境中安装依赖，再将下面的 `$AudioPython` 改成相应 Python 路径：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$AudioPython = ".\.venv\Scripts\python.exe"
```

安装 CUDA 版 PyTorch 请按 https://pytorch.org/get-started/locally/ 选择与设备匹配的安装命令。程序默认自动使用 CUDA，不可用时回退 CPU；可用 `--device cpu` 强制 CPU。不同设备/软件版本可能带来数值差异。

## 3. 数据准备与划分

程序需要的数据根目录结构为：

```text
UrbanSound8K/
  metadata/UrbanSound8K.csv
  audio/fold1/*.wav
  ...
  audio/fold10/*.wav
```

本地已包含数据，不需要再次下载或解压。需要手工指定位置时，用 `--data-root "实际包含 audio 和 metadata 的目录"`。程序验证 8732 条记录、类别映射、重复条目和全部音频路径，读取异常将报错，不静默跳过样本。

默认训练使用 fold1–8，验证用 fold9，测试用 fold10。**不进行音频切片随机划分。** 检查本地原始元数据时发现 5 个 `fsID` 跨官方 fold，因而采用更严格的原始录音隔离：完整保留测试 fold，验证集排除与测试同源的录音，训练集排除与验证/测试同源的录音。剔除只发生在有冲突的低优先级集合中，不改变测试集；具体见 `split_manifest.csv` 和 `config.json`。这属于“官方 fold + 同源清理”的协议，与不作清理的标准基准不完全相同。

特征标准化的均值和标准差仅在清理后的训练集上计算，并存入模型文件；验证、测试及预测均复用训练统计量。验证损失用于早停和选择最优模型，训练结束后才对测试集评估。

## 4. 快速执行

```powershell
# 单独提取特征，可选；train 会自动执行此步骤并复用缓存
& $AudioPython audio_classification.py prepare

# 默认正式实验：最多 60 轮，验证损失连续 12 轮未改善则停止
& $AudioPython -u audio_classification.py train

# 重新加载保存权重，在原测试 fold 上复核
& $AudioPython audio_classification.py evaluate --checkpoint outputs/lenet5/test_fold10/best.pt

# 对任一 WAV 推理（输出前三类 softmax 分数，不是校准后的置信度）
& $AudioPython audio_classification.py predict --checkpoint outputs/lenet5/test_fold10/best.pt --audio "音频路径.wav"
```

已有 `best.pt` 时程序拒绝覆盖。重复实验请指定新目录，例如：

```powershell
& $AudioPython audio_classification.py train --output outputs/lenet5_repeat --seed 42
```

训练参数可通过 `--epochs`、`--patience`、`--batch-size`、`--lr`、`--weight-decay`、`--seed`、`--threads` 修改。修改预处理代码后必须使用新的 `--cache` 目录或更新特征版本，不能复用旧特征。现有缓存检查预处理配置和元数据哈希，但不逐一哈希 WAV；如果替换了同名音频，也需要新的缓存目录。

可选运行完整十折（计算量约为单折十倍，当前已交付结果不是十折平均）：

```powershell
& $AudioPython -u audio_classification.py train --cross-validation --output outputs/lenet5_cv
```

外层依次以 fold1–10 测试；验证取循环前一 fold，其余训练，每折独立从零训练并做同源清理。程序写出 `cross_validation.json`，其中 `mean_accuracy` 是十个测试 fold 准确率的算术平均，`std_accuracy` 是总体标准差（ddof=0），不是把所有预测合并后的准确率。正式发表对比时需明确与官方标准协议的差异。

## 5. 模型与特征

预处理：WAV 整型幅值缩放到约 [-1, 1] → 多声道均值转单声道 → 多相滤波重采样到 16 kHz → 截断/尾部补零到 4 秒 → Hann 窗 STFT（512 点、256 点步长）→ 64 个 HTK 三角 Mel 滤波器的功率谱 → `10 log10(max(power, 1e-10))`，动态范围限制为 80 dB → 时间轴线性插值为 128 帧。每个 Mel 频带使用训练集均值/标准差标准化。没有使用数据增强。PCM/浮点 WAV 使用 SciPy 解码；遇到数据中的 DVI/IMA ADPCM WAV 时回退到 SoundFile，不跳过样本。

| 层 | 输出尺寸（不含 batch） |
| --- | --- |
| 输入 Log-Mel | 1 × 64 × 128 |
| 5×5 Conv，6 通道，Tanh | 6 × 60 × 124 |
| 2×2 平均池化 | 6 × 30 × 62 |
| 5×5 Conv，16 通道，Tanh | 16 × 26 × 58 |
| 2×2 平均池化 | 16 × 13 × 29 |
| Flatten + Linear + Tanh | 120 |
| Linear + Tanh | 84 |
| Linear（logits） | 10 |

这是 **LeNet-5 结构的教学复现/音频适配，不是 1998 年论文的逐位复刻**：保留两层 5×5 卷积、平均池化、Tanh 和 120→84→10 分类结构；输入由 32×32 灰度图改为 64×128 声谱图，C3 改为现代常见的全通道连接，池化不含可学习系数，C5 用覆盖整个剩余特征图的全连接层实现，输出用线性 logits + 交叉熵而非原论文的 RBF/欧氏距离分类器，优化器为 Adam。不能把本实验成绩称作原论文在 UrbanSound8K 上的成绩。

默认 Adam：学习率 0.001、权重衰减 0.0001、batch=64、随机种子 42；以验证交叉熵最低选模型。测试指标包括 accuracy、macro-F1、逐类 precision/recall/F1 和混淆矩阵。

## 6. 结果文件

`outputs/lenet5/test_fold10/` 中：

- `config.json`：设备、库版本、参数、数据数量、元数据 SHA-256。
- `split_manifest.csv`：每个样本的 train / validation / test / purged 归属。
- `history.json`、`training_curves.png`：每轮训练/验证损失和准确率。
- `best.pt`：最优权重、训练集标准化统计量、配置、最优轮次。
- `metrics.json`：测试准确率、macro-F1、逐类指标、混淆矩阵、训练时间。
- `test_predictions.csv`：逐条真实标签与预测标签。
- `confusion_matrix.png`：行是真实类，列是预测类。
- `logmel_example.png`：训练集声谱图示例。

只加载自己训练或可信来源的权重，程序使用 `weights_only=True`。单文件推理固定只读取前 4 秒；长录音连续检测需另做滑窗，不属于当前实验范围。

## 参考资料

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324. DOI: https://doi.org/10.1109/5.726791
2. Salamon, J., Jacoby, C., & Bello, J. P. (2014). *A Dataset and Taxonomy for Urban Sound Research*. ACM Multimedia. DOI: https://doi.org/10.1145/2647868.2655045
3. 数据集说明：https://urbansounddataset.weebly.com/urbansound8k.html ，以及本地 `UrbanSound8K_README.txt`。数据集及原始音频须遵守其随附许可和署名要求。
