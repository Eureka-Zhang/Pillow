"""
睡姿 5 分类模型（MLP + BatchNorm）：输入 (4, 256)，输出 5 类 logits。
"""
import torch
import torch.nn as nn

CLASS_NAMES = ["Middle", "Left-Back", "Right-Back", "Left", "Right"]
NUM_CLASSES = 5

INPUT_SIZE = 4 * 256  # 1024


class PostureClassifier(nn.Module):
    """MLP + BatchNorm：1024 -> hidden -> 5 classes"""

    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES, hidden=(256, 128), dropout=0.35):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        layers = []
        prev = input_size
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.head(self.backbone(x))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = PostureClassifier()
    x = torch.randn(2, 4, 256)
    y = model(x)
    print(y.shape)  # (2, 5)
    total, trainable = count_parameters(model)
    print(f"Total: {total:,}, Trainable: {trainable:,}")
