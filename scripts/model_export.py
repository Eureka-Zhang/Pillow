import torch
import os
from model import PostureClassifier

# 1. 加载模型
model = PostureClassifier()
ckpt_path = "posture_5class.pth"
if not os.path.exists(ckpt_path):
    # 尝试找 run1
    if os.path.exists("posture_5class_run1.pth"):
        ckpt_path = "posture_5class_run1.pth"

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# 2. dummy 输入 [B, 4, 256]
dummy = torch.randn(1, 4, 256)

# 3. 导出为 ONNX（固定 Batch=1，NPU 部署通常需要固定形状）
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=11,
    dynamic_axes=None,  # 改为 None，固定形状 [1, 4, 256]
)

print("ONNX 导出完成 -> model.onnx")
