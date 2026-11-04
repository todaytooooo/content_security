
## 网络结构

`1×28×28 → 固定补零至1×32×32 → Conv(1,6,5) → Tanh → AvgPool(2) → Conv(6,16,5) → Tanh → AvgPool(2) → Flatten(400) → FC(120) → Tanh → FC(84) → Tanh → FC(10)`。


采用常用 LeNet-5 分类改编版本：6/16 卷积通道、120/84 隐藏单元、Tanh 和平均池化。

## 文件职责与输出隔离

| 文件/目录 | 作用 |
| --- | --- |
| `FGSM/fgsm_mnist.py` | LeNet 定义、共享训练函数、FGSM 攻击与评估 |
| `FGSM/test_lenet.py` | 检查结构、参数数量、形状、输入梯度及权重加载 |
| `FGSM/outputs_lenet/` | LeNet 基线权重 `mnist_lenet.pt`、FGSM 指标和样例 |
| `PGD/pgd_mnist.py` | 加载该 LeNet 基线，比较 FGSM 和 PGD；`--train` 可另训 LeNet |
| `PGD/outputs_lenet/` | LeNet 的攻击指标、对比曲线和 PGD 样例 |
| `AdversarialTraining/adversarial_training.py` | 基线、干净微调、固定 FGSM 增强三组对照 |
| `AdversarialTraining/outputs_lenet/` | 固定增强实验权重、指标和对比图 |
| `AdversarialTraining/online_rs_fgsm.py` | 独立划分、重训 LeNet 共同基线、逐轮 PGD 验证及在线 RS-FGSM 对照 |
| `AdversarialTraining/outputs_online_lenet/` | 划分索引、逐轮记录、选择结果、最佳/末轮权重及测试指标 |
| `run_lenet_experiments.py` | 按依赖顺序运行上述四个完整实验 |


## 运行

在对应子目录依次执行：

```powershell
python fgsm_mnist.py
python pgd_mnist.py
python adversarial_training.py
python online_rs_fgsm.py
```

第一个基线使用完整 60,000 张训练图片；在线实验为避免验证数据泄漏，另在 55,000/5,000 划分上从头训练共同基线，不加载完整训练集的基线权重。测试仍为官方 10,000 张图片。


