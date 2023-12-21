# HW2-Classification

## Description

- Phoneme 分类任务
- Training data: 3429 个预处理后的音频 (2116794 frames)
- Testing data: 857 个预处理后的音频 (527364 frames)
- Label: 41 个类别, 每一个类代表一个 phoneme

`libriphone` 文件夹下各文件简介：

- `test_split.txt` : 测试数据集的 feature id
- `train_labels.txt` : 训练数据集的 feature id 和其对应 labels
- `train_split.txt` : 训练数据集的 feature id
- `feat/test/` : 测试数据集的 features。每个 feature 保存为 Tensor，文件名为 "{feature_id} + .pt"
- `feat/train/` : 训练数据集的 features。每个 feature 保存为 Tensor，文件名为 "{feature_id} + .pt"

## Baseline

|        | Public Baseline | Hints                         | Training Time |
|--------|-----------------|-------------------------------|---------------|
| Simple | 0.49798         | sample code                   | ~30 min       |
| Medium | 0.66440         | concat n frames, add layers   | 1~2 h         |
| Strong | 0.74944         | batch norm, dropout           | 3~4 h         |
| Boss   | 0.83017         | sequence-labeling (using RNN) | 6~ h          |

## Strategy

由于每个 frame 只有 25 ms 的语音，单个 frame 不太可能表示一个完整的 phoneme：

- 通常，一个 phoneme 会跨越多个 frame
- 连接相邻的 phoneme 用于训练
