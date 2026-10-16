# HOG + SVM 行人目标检测

本实验使用 OpenCV 内置的预训练 HOG（方向梯度直方图）+ 线性 SVM 行人检测器，对一张场景图直接检测并画出行人框。

## 运行

先激活课程环境：

```powershell
conda activate content-security
```

在本目录运行最终检测：

```powershell
python final_hog_detector.py --image data/test/example.png --output outputs/pedestrian_detection_final.png
```

输出文件为 `outputs/pedestrian_detection_final.png`。该图片来自 INRIA Person Dataset 的测试集，结果框出 1 名完整行人，可直接作为实验结果截图。

唯一运行入口是 `final_hog_detector.py`；无需训练或加载 `.joblib` 模型。课程环境已安装依赖；如需重装，在激活环境后执行 `python -m pip install -r requirements.txt`。

目录中 `data/test` 保存示例图和你自己的测试图，`outputs` 保存检测结果。`note.md` 为保留的个人笔记，涉及旧训练流程，不是当前运行步骤。

## 算法流程

```text
输入场景图 -> HOG 特征提取 -> 预训练线性 SVM 分类 -> 多尺度滑动窗口 -> 输出行人框
```

HOG 将局部梯度方向统计为特征，适合描述人体轮廓；线性 SVM 判断每个滑动窗口是否为行人。预训练检测器由 OpenCV 提供，因此不需要自行下载、裁剪或训练正负样本。

## 更换图片

将 `--image` 替换为待检测图片路径即可。例如：

```powershell
python final_hog_detector.py --image "你的图片.jpg" --output outputs/result.jpg
```

复杂拥挤、严重遮挡的图像可能出现漏检或误检，这是传统 HOG+SVM 方法的局限。实验演示优先选取行人完整、无遮挡、与背景对比清晰的场景。
