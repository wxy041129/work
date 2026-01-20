# 多模态情感分类实验（实验五）

本项目实现了基于图文对的多模态情感分类模型，支持消融实验（文本-only / 图像-only / 多模态融合），并生成测试集预测结果。

##  环境依赖

请使用 Python 3.8+ 并创建虚拟环境：

```bash
pip install -r requirements.txt
```

注意：CLIP 需从官方 GitHub 安装（已在 `requirements.txt` 中指定）。

##  项目结构

```
.
├── data/                     # 原始数据目录（不上传）
├── train.txt                 # 训练集标签文件
├── test_without_label.txt    # 测试集
├── requirements.txt          # 依赖库列表
├── dataset.py                # 自定义多模态数据集类
├── model.py                  # 多模态分类模型（支持三种模式）
├── utils.py                  # 工具函数（如划分验证集）
├── train.py                  # 训练脚本（自动进行消融实验）
├── predict.py                # 生成测试集预测结果
└── README.md                 # 本说明文件
```

##  运行步骤

### 1. 训练模型（含消融实验）

运行以下命令，将自动训练三种模型并保存最佳权重：

```bash
python train.py
```

输出文件：
- `best_model_concat.pth`      → 多模态融合模型
- `best_model_text_only.pth`   → 仅文本模型
- `best_model_image_only.pth`  → 仅图像模型

### 2. 生成测试集预测结果

使用多模态模型对测试集进行预测：

```bash
python predict.py
```

输出文件：
- `prediction.txt`：格式与 `test_without_label.txt` 一致，但 `null` 已替换为预测标签（`positive` / `neutral` / `negative`）

##  模型设计说明

- **Backbone**：采用 OpenAI 的 CLIP（ViT-B/32）作为图像和文本编码器
- **融合策略**：特征拼接（concatenation）后接 MLP 分类头
- **消融实验**：通过切换 `fusion_type` 参数实现三种模式对比
- **训练细节**：冻结 CLIP 主干，仅微调分类头；使用 AdamW 优化器，学习率 `1e-4`
