# 在 MNIST 上复现 FGSM

参考：Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy，*Explaining and Harnessing Adversarial Examples*，ICLR 2015。论文：https://arxiv.org/abs/1412.6572 。

本实验复现论文的 FGSM 核心算法及对抗样本现象。使用自行定义的 CNN，不声称逐项复现论文的模型、超参数或数值结果；本实验也没有实现论文中的对抗训练。这里的分类准确率评估不是对抗样本检测器。

## 1. 算法理解

令模型参数为 θ，输入为 x，真实标签为 y，分类损失为 J(θ,x,y)。非定向攻击希望在有限扰动下增大真实标签对应的损失：

$$\max_{\|\delta\|_\infty\le\epsilon} J(\theta,x+\delta,y).$$

对输入作一阶近似：

$$J(\theta,x+\delta,y)\approx J(\theta,x,y)+\delta^T\nabla_xJ(\theta,x,y).$$

在每个像素均受 |δᵢ|≤ε 约束时，线性项的最大值由梯度符号方向取得，因此：

$$x_{adv}=\operatorname{clip}_{[0,1]}\left(x+\epsilon\operatorname{sign}(\nabla_xJ(\theta,x,y))\right).$$

FGSM 是白盒、单步、非定向攻击：需要模型输入梯度，只计算一次梯度，不要求攻击后变成某个指定数字。`model.eval()` 只切换模型行为，不会关闭梯度；生成攻击时不能使用 `torch.no_grad()`。代码用 `torch.autograd.grad` 只求输入梯度，不更新模型参数。

ε 表示原始 [0,1] 像素空间中的最大改动。例如 ε=0.1 对应 8 位灰度值约 25.5。裁剪后实际改动可能小于 ε。脚本只做 `ToTensor()`，不额外标准化，以免把标准化空间的 ε 和原始像素空间的 ε 混用。

高维输入中，每个分量的小幅改变可以在线性项中累积，解释了视觉变化较小却可能改变分类的现象。一阶近似不保证对任意模型、样本都攻击成功，也不保证实测准确率随 ε 严格单调下降。

## 2. 安装与运行

在本目录打开 PowerShell，使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fgsm_mnist.py
```

首次运行自动下载 MNIST，需要网络。默认使用全部 60,000 张训练图和 10,000 张测试图，Adam 学习率 0.001，训练 5 轮，批量大小 128，随机种子 42，有 CUDA 时自动使用 GPU。数据和输出默认位于脚本目录下，不受启动时的工作目录影响。PyTorch 与 torchvision 应安装相互兼容的版本；特定 CUDA 平台可参考 https://pytorch.org/get-started/locally/ 安装。

先检查流程能否跑通（子集结果不能作为完整实验结果）：

```powershell
.\.venv\Scripts\python.exe fgsm_mnist.py --epochs 1 --train-limit 2000 --test-limit 256 --device cpu --output-dir outputs_smoke
```

训练并评估完整数据集：

```powershell
.\.venv\Scripts\python.exe fgsm_mnist.py --epochs 5 --epsilons 0 0.05 0.1 0.15 0.2 0.25 0.3
```

复用已训练模型，仅重新攻击和评估：

```powershell
.\.venv\Scripts\python.exe fgsm_mnist.py --checkpoint outputs/mnist_cnn.pt --output-dir outputs_eval
```

相同输出目录会覆盖已有实验结果。随机种子和确定性 cuDNN 设置可减少差异，但跨设备、跨依赖版本不保证数值完全一致。

## 3. 结果与指标

`outputs/` 中生成：

| 文件 | 内容 |
| --- | --- |
| `mnist_cnn.pt` | 本次训练的模型权重；仅加载权重评估时不另存 |
| `metrics.csv` | 每个 ε 对应的指标，比例范围为 [0,1] |
| `results.json` | 参数、设备、PyTorch 版本、训练记录和指标 |
| `accuracy_vs_epsilon.png` | ε 与全部测试样本对抗准确率的关系 |
| `examples_eps_*.png` | 原图、放大显示的实际扰动、对抗图及预测 |

三个指标的分母要区分：

- 干净准确率：原图分类正确数量 / 全部测试样本数量。
- 对抗准确率：对全部测试图片生成攻击后的正确数量 / 全部测试样本数量。
- 攻击成功率：原本正确且攻击后错误的数量 / 原本正确的数量。原本没有正确样本时，该值写为 `null`（CSV 中为空）。

每个 ε 都从同一批原图生成攻击，不把上一次的对抗图片作为下一次输入。相同输入梯度可供多个 ε 复用。ε=0 始终纳入评估，并检查图像不变、准确率与干净输入一致；同时检查扰动 L∞ 上界和数值有限性。

示例图优先展示原本正确、攻击成功的样本，属于有选择的展示，不能用它代替全量指标。若某个 ε 没有成功攻击，则不生成该 ε 的示例图。扰动图使用按 ε 缩放的红蓝色阶，便于观察方向，不能把色彩强度当作对抗图的实际视觉变化。

本实现准备过程中本地执行工具启动失败，尚未实际训练或验证代码运行。因此此处不提供未经运行的准确率、成功率或图表；执行上述命令后，以生成文件为准。

## 4. 分析建议

先确认完整测试集上的干净准确率，再比较不同 ε 的对抗准确率与攻击成功率。通常增大 ε 会使攻击更明显、更容易奏效，但要结合实际曲线判断。ε 很大时图片可能已出现明显失真，不宜将其称为不可察觉扰动。

如需进一步研究论文中的防御思路，可将干净样本和对抗样本共同用于训练，再在相同攻击设置下比较两个模型；单独的 FGSM 评估不足以证明模型对其他攻击也鲁棒。
