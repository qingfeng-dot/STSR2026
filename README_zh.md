# STSR Task 2：三维牙科配准

[English README](./README.md)

本仓库实现了当前用于 STSR Task 2 的完整流程：将上颌、下颌口内扫描（STL）分别配准到 CBCT 体数据（`.nii.gz`），并为每个牙颌预测一个 4 x 4 变换矩阵。

当前模型采用双分支 PointNet++ 编码器、双向交叉注意力、双 Softmax 对应点匹配和加权 SVD/Procrustes 配准头。上颌与下颌使用同一训练脚本独立训练；整个流程还支持可选的对比学习预训练、可复用点云缓存、内部验证、测试时增强（TTA）以及提交文件自动打包。

## 项目结构

```text
.
├── configs/
│   ├── pretrain_config.yaml     # 可选的对比学习预训练配置
│   ├── train_config.yaml        # 有监督配准训练配置
│   └── inference_config.yaml    # 批量推理与提交配置
├── data/                        # 数据集、预处理与增强
├── losses/                      # 对比学习与配准损失
├── models/                      # PointNet++、交叉注意力与 SVD 配准头
├── utils/                       # 变换与仿射坐标域工具
├── main_pretrain.py             # 可选的编码器预训练
├── main_train.py                # 上颌/下颌训练
├── main_inference.py            # 推理与 ZIP 打包
├── eval_local.py                # 本地平移/旋转误差评测
└── requirement.txt
```

## 环境安装

推荐使用 Python 3.9 和支持 CUDA 的 GPU。请先安装与本机 CUDA 版本匹配的 PyTorch，再安装其余依赖：

```bash
conda create -n stsr-task2 python=3.9 -y
conda activate stsr-task2

# 如果尚未安装，请先安装与 CUDA 匹配的 PyTorch：
# https://pytorch.org/get-started/locally/
pip install -r requirement.txt
```

`torch_geometric` 依赖 `torch-cluster` 等编译扩展。如果环境无法自动安装兼容版本，请参考 [PyTorch Geometric 安装文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)，选择与 PyTorch 和 CUDA 匹配的 wheel。

## 数据目录

有标签训练数据应同时包含 `Images` 和 `Labels`（也兼容小写目录名）：

```text
Train-Labeled/
├── Images/
│   └── 001/
│       ├── CBCT.nii.gz
│       ├── upper.stl
│       └── lower.stl
└── Labels/
    └── 001/
        ├── upper_gt.npy
        └── lower_gt.npy
```

每个标签都是形状为 `(4, 4)` 的 NumPy 变换矩阵。无标签预训练数据使用相同的 `Images/<病例编号>/...` 结构，但不需要 `Labels`。推理时，`inference_data_root` 应直接指向包含病例子目录的目录：

```text
Validation/Images/
└── 001/
    ├── CBCT.nii.gz
    ├── upper.stl
    └── lower.stl
```

## 运行流程

### 1. 修改配置路径

根据本机数据与输出位置修改：

- `configs/pretrain_config.yaml`：可选的自监督预训练；
- `configs/train_config.yaml`：有监督配准训练；
- `configs/inference_config.yaml`：权重加载、预测输出与提交 ZIP。

训练和推理配置中的模型结构与数据预处理参数必须保持一致，尤其是点数、`feature_dim`、`target_dense_points`、`registration_head`、CBCT 阈值/ROI 以及平移残差相关设置。

### 2. 可选：对比学习预训练

在 `configs/pretrain_config.yaml` 中设置 `jaw_type` 和 `experiment_name`，然后分别预训练上颌、下颌编码器：

```bash
python main_pretrain.py
```

最佳预训练权重保存在：

```text
<output_dir>/<experiment_name>/checkpoints/best_model.pth
```

如需加载预训练权重，在 `configs/train_config.yaml` 中设置 `load_pretrained: true`，并更新 `pretrained_checkpoints`。预训练不是必需步骤；若希望从头训练，可设置 `load_pretrained: false`。

### 3. 分别训练上颌和下颌

命令行参数会覆盖 YAML 中的 `jaw_type` 和实验名称：

```bash
python main_train.py --config configs/train_config.yaml --jaw upper
python main_train.py --config configs/train_config.yaml --jaw lower
```

默认分别使用 `experiment_names.upper` 和 `experiment_names.lower` 中的名称；也可以通过 `--experiment-name` 指定其他名称。训练结果保存在：

```text
<output_dir>/<experiment_name>/
├── checkpoints/
│   ├── best_model.pth
│   └── latest_model.pth
├── logs/
└── log.txt
```

如果配置的验证集没有标签，当前配置会从有标签训练集按病例划分固定的、考虑手性分布的内部验证集。启用点云缓存后，首次运行会在 `point_cache_dir` 下建立 STL/CBCT 缓存，后续训练会直接复用。

可使用 TensorBoard 同时查看两个实验：

```bash
tensorboard --logdir ./experiments22
```

### 4. 推理并生成提交文件

在 `configs/inference_config.yaml` 中设置 `inference_data_root`、`output_dir`、上/下颌实验名称（或显式权重路径）、`prediction_root` 和 `submission_zip_path`，然后运行：

```bash
python main_inference.py --config configs/inference_config.yaml
```

脚本会加载上、下颌模型，从 CBCT 中提取阈值表面点，使用与训练一致的归一化和分牙颌 ROI 设置，聚合 TTA 预测，并同时生成预测目录和提交 ZIP：

```text
prediction_root/
└── 001/
    ├── upper_gt.npy
    └── lower_gt.npy
```

推理配置还提供 ICP 精配准、分指标权重组合和仿射坐标域修正等可选功能；当前默认竞赛配置中这些功能处于关闭状态。

### 5. 本地评测

对于有标签数据，可以计算平均平移误差（毫米）和旋转误差（度）：

```bash
python eval_local.py \
  --prediction_root ./prediction_output \
  --label_root /path/to/Validation/Labels \
  --jaw both
```

如只评测一个牙颌，可将 `--jaw` 改为 `upper` 或 `lower`。

## 注意事项

- 请从仓库根目录运行所有命令，确保配置中的相对路径能够正确解析。
- 上颌和下颌虽然共享模型结构与训练代码，但使用彼此独立的权重。
- `main_inference.py` 启动时会重新创建 `prediction_root`，不要将其指向需要保留其他文件的目录。
- `best_model.pth` 用于推理，`latest_model.pth` 保存最近一次训练状态。
