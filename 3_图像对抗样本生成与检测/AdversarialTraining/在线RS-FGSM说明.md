# 固定增强与在线 RS-FGSM 对照

完整实验已完成，见 [结果分析](在线RS-FGSM结果分析.md)。在线组测试正常准确率 99.23%，ε=0.2 的 40 步、3 次重启 PGD 后准确率为 81.07%；逐轮验证未见持续崩溃。

脚本：`online_rs_fgsm.py`。新增输出保存在 `outputs_online/`，原固定增强实验代码和结果保留。

## 数据与公平对照

先以种子 42 将 MNIST 官方训练集随机分为 55,000 张训练图片、5,000 张验证图片，索引保存在 `split_indices.json`。从随机权重在这 55,000 张图片上干净训练 5 轮，得到本次共同基线。不能直接使用之前的模型：它见过全部 60,000 张图片，会污染独立验证集。

从同一基线分别继续训练 5 轮：

- `CleanControl`：正常图片重复一份，作为额外训练对照。
- `FixedFGSM`：正常图片与基线生成的固定 FGSM 样本按 1:1 混合。
- `OnlineRSFGSM`：每个批次针对当前模型生成 RS-FGSM，正常与对抗图片按 1:1 混合。

各组使用相同训练图片顺序、初始权重、Adam 学习率 0.001、训练轮次、每批 128 张原图（混合后 256 张）及参数更新次数。没有训练时数据增强；在线攻击额外消耗计算。

## 在线攻击

先生成随机起点 `z=clip(x+Uniform(-ε,ε),0,1)`，在 z 上计算真实标签交叉熵的输入梯度，再更新 `z+α sign(grad)`，最后投影到 `[x-ε,x+ε]∩[0,1]`。默认 ε=0.2、α=0.25（1.25ε）。注意修改 ε 不会自动改变显式的 `--rs-alpha` 参数。

每次更新模型前重新生成攻击，攻击不更新模型参数，结果与计算图分离后用于训练。目标为 0.5 倍正常损失加 0.5 倍对抗损失。该实现参考随机起点单步训练思路（Wong 等，*Fast is better than free: Revisiting adversarial training*，https://arxiv.org/abs/2001.03994），不是原论文所有超参数和训练技巧的逐项复现。

## 逐轮监控与选模型

每轮在全部 5,000 张独立验证图片上测正常准确率、FGSM 准确率和 PGD 准确率。验证 PGD 默认 ε=0.2、40 步、α=0.01、1 次随机起点；各轮固定评估随机种子，并恢复训练的随机状态。记录包含共同基线 epoch=0。

根据验证 PGD 准确率严格提高时保存 `*_best.pt`，相等时保留更早的模型。epoch=0 也是候选，因此增强训练全部退化时最佳模型可以是原始基线。另存 `*_last.pt`，保留最后一轮以观察退化。训练固定运行 5 轮，不根据测试集选轮次，也不在首次退化时停止，便于观察曲线。

所有选择完成后，才使用官方 10,000 张测试图片。报告基线、各组最佳及最后一轮的正常/FGSM/PGD 指标。测试默认 ε=0.1、0.2、0.3，PGD 40 步、3 次随机起点。PGD 准确率采用保守计数：正常输入和每次重启的最终攻击都分类正确才算存活。没有逐步保存成功攻击，也未提供鲁棒性认证，因此有限攻击仍可能漏掉对抗样本。

曲线突然下降可作为灾难性过拟合的线索，但不能仅凭一次最终准确率就确认具体机制。固定增强也可能退化，两种训练过程应分别描述。

## 运行

在本目录、已有可用环境中执行：

```powershell
conda activate content-security
python -m unittest -v
python online_rs_fgsm.py
```

快速流程检查（单独输出，不作为完整实验结论）：

```powershell
python online_rs_fgsm.py --baseline-epochs 1 --epochs 1 --validation-size 32 --train-limit 64 --test-limit 32 --pgd-steps 2 --test-restarts 1 --test-epsilons 0.2 --output-dir outputs_online_smoke
```

输出包括配置、划分索引、基线/最佳/最后权重、`validation_history.json/csv`、`selection.json`、`test_metrics.csv`、`results.json` 和 `validation_curves.png`。默认参数预先固定，单次实验不代替多随机种子统计。相同目录重复运行会覆盖同名文件。
