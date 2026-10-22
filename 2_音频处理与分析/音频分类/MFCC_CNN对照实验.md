# MFCC + 2D CNN 对照实验

实验日期：2026-10-22。本次试验固定数据划分和训练协议，只执行一次预先确定的 MFCC + 轻量 2D CNN 配置；验证集用于选择最低验证损失的权重，fold10 只在训练结束后评估一次。

## 设置

- 划分：官方 fold1--8 训练、fold9 验证、fold10 测试；保留完整测试集，并清理训练/验证集中与高优先级集合相同 `fsID` 的样本。因此为训练 7000、验证 816、测试 837，另有 79 条 purged。
- 特征：音频重采样为 16 kHz，截断或补零到 4 s；512 点 Hann STFT、256 点 hop、64 个 HTK Mel 频带、80 dB 动态范围，并插值到 128 帧。对每帧 64 维 Log-Mel 做正交 DCT-II，保留前 40 个系数，得到 `1 x 40 x 128` 的 MFCC。标准化统计量只由训练集计算。
- 模型：`Conv(1,16,3) -> BN -> ReLU -> MaxPool(2) -> Conv(16,32,3) -> BN -> ReLU -> MaxPool(2) -> AdaptiveAvgPool2d(1) -> Dropout(0.3) -> Linear(32,10)`，共 5,226 个参数。
- 训练：Adam，学习率 0.001，weight decay 0.0001，batch size 64，seed 42，最多 60 epoch，patience 12。设备为 RTX 4060 Laptop GPU。

## 结果

| 指标 | MFCC + 2D CNN |
| --- | ---: |
| 实际训练轮数 | 60（达到上限） |
| 最低验证损失对应轮次 | 56 |
| 最低验证损失 | 1.0886 |
| 该轮验证准确率 | 63.60% |
| 测试 Accuracy | **61.89%**（518/837） |
| 测试 Macro-F1 | **60.82%** |
| 测试交叉熵 | 1.1189 |

与此前最佳的 Log-Mel + LeNet GAP 对照（51.37% accuracy、50.38% macro-F1）相比，本次固定 fold10 上分别高 10.51 和 10.44 个百分点。这个数值只适用于当前单种子、单测试 fold 的探索性比较，不能据此宣称稳定的总体优势，也不应继续围绕 fold10 调参。

这不是只改分类头的严格消融：MFCC 改变了输入表征，CNN 同时改用了 BatchNorm、ReLU、MaxPool 和全局池化。MFCC 系数轴也不是物理频率轴，因此二维卷积沿该轴的效果应作为工程上的模型比较，而非频率局部模式的直接物理解释。要验证结论，需要先固定方案，再运行多 fold 或多随机种子实验。

## 复现

在本目录执行：

```powershell
$AudioPython = "D:\Tools\Anaconda\envs\content-security\python.exe"
& $AudioPython -m unittest -v test_audio_classification.py
& $AudioPython -u audio_classification.py train --model cnn --feature mfcc --output outputs/mfcc_cnn_repeat
& $AudioPython audio_classification.py evaluate --checkpoint outputs/mfcc_cnn_repeat/test_fold10/best.pt
```

已存在 `best.pt` 时训练命令会拒绝覆盖。当前一次实验的完整产物位于 `outputs/mfcc_cnn/test_fold10/`，包括配置、划分清单、最佳权重、训练历史、指标、逐样本预测、训练曲线和混淆矩阵。

## 三随机种子稳定性验证

在原方案不变的前提下，预先指定新增 seed 7、2026，与已有 seed 42 组成三次独立训练。没有依据测试结果修改结构、超参数或挑选测试最优种子；每次仍按最低 fold9 验证损失选择权重，训练结束后评估 fold10。三次均训练到 60 轮上限。

| Seed | 最佳验证轮次 | 测试 Accuracy | 测试 Macro-F1 |
| --- | ---: | ---: | ---: |
| 42 | 56 | 61.89% | 60.82% |
| 7 | 58 | 63.20% | 63.27% |
| 2026 | 59 | 60.33% | 59.86% |
| 均值 ± 样本标准差 | — | **61.81% ± 1.44 个百分点** | **61.32% ± 1.76 个百分点** |

这里标准差使用 ddof=1，描述不同训练种子的波动，不是置信区间。Accuracy 范围为 60.33%--63.20%。三次都在同一测试 fold 上评估，测试样本没有增加；结果支持当前固定划分下训练表现较稳定，不能证明跨录音场景或跨 fold 的泛化性能。此前 Log-Mel 对照只有一个种子，也不能据此进行可靠的显著性检验。

新增产物分别位于 `outputs/mfcc_cnn_seed7/test_fold10/`、`outputs/mfcc_cnn_seed2026/test_fold10/`。`summarize_mfcc_seeds.py` 核对三次划分清单的 SHA-256 完全一致、主要设置与环境一致，并生成 `outputs/mfcc_seed_validation/summary.json` 和 `runs.csv`。

```powershell
& $AudioPython -u audio_classification.py train --model cnn --feature mfcc --seed 7 --output outputs/mfcc_cnn_seed7_repeat
& $AudioPython -u audio_classification.py train --model cnn --feature mfcc --seed 2026 --output outputs/mfcc_cnn_seed2026_repeat
# 汇总本次保存的三个固定路径；重复实验请相应修改 RUNS
& $AudioPython summarize_mfcc_seeds.py
```
