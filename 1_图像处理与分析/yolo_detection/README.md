# 实验任务 2：YOLOv8 本地图片目标检测

使用课程公共 Conda 环境 `content-security`，部署 Ultralytics YOLOv8n 的 COCO 预训练模型，在 COCO128 的 128 张图片上运行目标检测和标注验证。无需从零训练即可完成本任务。默认优先使用 NVIDIA GPU，也支持 `--device cpu`。

## 运行方法

在 PowerShell 中进入本文件所在目录：

```powershell
conda activate content-security
cd 'D:\2026-2027大三上\content_security\1_图像处理与分析\yolo_detection'
python yolo_experiment.py all
```

首次运行会下载模型和数据集；以后使用本地文件。各步骤也可单独运行：

```powershell
python yolo_experiment.py prepare
python yolo_experiment.py info
python yolo_experiment.py predict
python yolo_experiment.py validate
python yolo_experiment.py predict --source '你的图片.jpg'
python yolo_experiment.py predict --source '你的图片目录' --conf 0.4
```

更换输入图片后，检测 JSON 会被本次结果替换，`outputs/predict` 中同名图片也会覆盖；目录中可能保留上次的其他图片。每次验证使用固定的 `outputs/val` 目录。

重新安装环境时（已有环境直接运行，不要重建）：

```powershell
conda activate content-security
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

保留 `numpy<2`、`opencv-python<4.12`，兼容当前课程中 HOG/SVM/KNN 的环境。实际安装版本记录在 `outputs/environment-lock.txt`。

## 结果文件

| 路径 | 内容 |
| --- | --- |
| `models/yolov8n.pt` | 本地预训练权重 |
| `data/coco128/images/train2017` | 128 张原始图片 |
| `data/coco128/labels/train2017` | YOLO 格式人工标注 |
| `data/coco128_local.yaml` | 本地数据集路径及 80 个类别名 |
| `outputs/predict/*.jpg` | 绘制类别、置信度及边界框的检测图片 |
| `outputs/predict/labels/*.txt` | 预测框：类别、归一化中心坐标/宽高、置信度 |
| `outputs/detections.json` | 每张图片的类别、置信度、像素坐标及耗时 |
| `outputs/prediction_summary.json` | 图片数、检测框数、各类检测数量 |
| `outputs/val` | 验证图片、混淆矩阵、PR/F1 曲线等 |
| `outputs/metrics.json` | 实际 Precision、Recall、mAP50、mAP50-95 |
| `outputs/model_structure.txt` | 本地模型模块结构、参数量及模型配置 |
| `outputs/environment.json` | Python、PyTorch、CUDA、GPU、Ultralytics 信息 |

## 数据和指标的含义

COCO128 是 COCO train2017 的小型演示子集，含 128 张图片，采用 COCO 的 80 类标签体系。标注文件每行是 `class x_center y_center width height`，坐标相对图像宽高归一化到 0–1。选它是为了快速完成多类别检测及有标注的结果验证。

本实验使用 COCO 预训练权重，没有执行训练或微调。COCO128 来自 COCO 训练集，配置中 train/val 使用同一图片目录，因此这里的指标仅用于验证部署、分析示例检测效果，不能声称是独立测试集泛化性能；也不能当作完整 COCO val2017 的官方性能。需要独立评估时可另下载 COCO val2017。

推理展示采用置信度 0.25、输入尺寸 640；验证使用框架默认的低置信度阈值生成 PR 曲线，因此验证统计不等于展示框数。Precision 表示预测中正确的比例，Recall 表示真实目标被找到的比例；mAP50 使用 IoU 0.5，mAP50-95 汇总 IoU 0.5 至 0.95，后者对定位更严格。实际数值见 `metrics.json`。

## YOLOv8 结构及创新点

YOLO 是单阶段检测器，一次前向传播直接预测多类别目标及其边界框。YOLOv8n 中的 n（nano）表示较小的模型规模。

1. **Backbone：Conv、C2f、SPPF。** 卷积逐步降低空间分辨率并提取特征；C2f 通过分支、Bottleneck 和拼接改善梯度流及特征复用；SPPF 通过连续池化聚合不同感受野的上下文。
2. **Neck：上采样、Concat、C2f。** 使用 FPN/PAN 风格融合深层语义与浅层位置细节，生成 P3/P4/P5 三个尺度的检测特征，覆盖小、中、大目标。
3. **Head：解耦且 anchor-free。** 分类和边界框回归使用不同分支，以特征点预测目标，不需要预设多种 anchor box 尺寸。YOLOv8 检测头没有独立的 objectness 分支。
4. **训练改进。** 使用 Task-Aligned Assigner 分配正样本，结合分类损失、CIoU 定位损失和 DFL（Distribution Focal Loss）学习边界框分布。它们解释预训练模型如何得到；本次实验只使用训练好的权重。
5. **推理后处理。** 根据类别分数过滤候选框，再通过 NMS 抑制高度重叠的重复框，最终将框映射回原图。

相比 YOLOv5，YOLOv8 的主要变化包括 C2f、anchor-free 解耦检测头、样本分配和损失设计。SPPF、多尺度融合、NMS 属于沿用或组合的机制，不应全部写成 YOLOv8 首创。YOLOv8 本身也不是完全免 NMS 的模型。

结构可以通过 `python yolo_experiment.py info` 查看，与 `outputs/model_structure.txt` 对照理解。预训练检测可识别 COCO 定义的类别；对于体系以外的类别，需要相应数据和训练。

## 本机实测结果

在 `content-security` 中使用 Python 3.9.25、Ultralytics 8.4.177、PyTorch 2.6.0+cu126，成功调用 RTX 4060 Laptop GPU。YOLOv8n 原始模型有 3,157,200 个参数，输入尺寸为 640。

128 张图片全部生成了带框结果；展示阈值 0.25 下共得到 642 个预测框，其中 199 个预测为 person。这些是模型预测数量，不是人工确认的正确目标数量。

| 验证指标 | 实测值 |
| --- | --- |
| Precision | 62.90% |
| Recall | 52.90% |
| mAP@0.5 | 60.20% |
| mAP@0.5:0.95 | 44.34% |
| 验证平均前向推理耗时 | 3.16 ms/图片 |

耗时来自 GPU batch=8 验证，不包含完整的磁盘读取、输出绘图和首次初始化耗时，不应直接换算成端到端实时帧率。指标的适用范围见上面的 COCO128 说明。

已检查 `outputs/predict/000000000074.jpg`（街景中的人、自行车、狗）和 `outputs/val/val_batch0_pred.jpg`（多场景验证拼图）。可见多类别检测能够工作，同时仍有小目标漏检、标签重叠及误检，例如拼图中的气球被识别为 wine glass，适合在实验分析中讨论局限。

官方参考：

- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com/datasets/detect/coco128/
- https://docs.ultralytics.com/modes/predict/
- https://docs.ultralytics.com/modes/val/

这里只完成实验任务 2 的部署、检测、结果验证及结构说明，没有填充课程实验报告模板。
