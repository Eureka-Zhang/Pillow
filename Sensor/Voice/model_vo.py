import torch
import torch.nn as nn
import torch.onnx


# =========================
# TCN Block（因果 + 空洞）
# =========================
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.padding = padding

    def forward(self, x):
        y = self.conv(x)
        y = y[:, :, :-self.padding]   # causal cut
        return self.relu(y + self.res(x))


# =========================
# CNN + TCN + Dual Head
# =========================
class CNN_TCN_MTL(nn.Module):
    def __init__(self, snore_classes = 2, posture_classes=6):
        super().__init__()

        # -------- CNN --------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(8, 2), padding=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=(8, 2), padding=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(8, 2), padding=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # -------- TCN --------
        self.tcn = nn.Sequential(
            TCNBlock(64, 64, dilation=1),
            TCNBlock(64, 64, dilation=2),
            TCNBlock(64, 64, dilation=4),
            TCNBlock(64, 64, dilation=8)
        )

        # -------- Heads --------
        self.snore_head = nn.Linear(64, snore_classes)              # snore / non-snore
        self.posture_head = nn.Linear(64, posture_classes)

    def forward(self, x):
        """
        x: [B, 1, F, T]
        """
        x = self.cnn(x)            # [B, 64, F', T']
        x = x.mean(dim=2)          # [B, 64, T']
        x = self.tcn(x)            # [B, 64, T']
        feat = x.mean(dim=2)       # [B, 64]

        snore_logits = self.snore_head(feat)
        posture_logits = self.posture_head(feat)

        return snore_logits, posture_logits


# =========================
# 参数统计
# =========================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =========================
# ONNX 导出
# =========================
if __name__ == "__main__":
    model = CNN_TCN_MTL(posture_classes=6)
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 300)

    torch.onnx.export(
        model,
        dummy_input,
        "cnn_tcn_mtl.onnx",
        input_names=["input"],
        output_names=["snore_logits", "posture_logits"],
        opset_version=11
    )

    print("✅ Model converted to ONNX and saved as cnn_tcn_mtl.onnx")

    total, trainable = count_parameters(model)
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
