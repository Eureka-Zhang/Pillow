from rknn.api import RKNN
import random
import numpy as np
from pathlib import Path
import shutil

# 复用 train.py 的数据加载逻辑
from train import load_all_data

data_dir = "data"

print("开始加载数据集...")
# load_all_data 返回的是归一化后的 numpy 数组 X, Y
# X: (N, 4, 256), Y: (N,)
X, Y = load_all_data(data_dir)
print(f"数据集加载完成，共 {len(X)} 条样本")

# 采样数量（可按需要调整）
NUM_SAMPLES = 800

# 随机采样
print("开始随机采样...")
total_samples = len(X)
idxs = random.sample(range(total_samples), min(NUM_SAMPLES, total_samples))
print(f"随机采样完成，采样数量: {len(idxs)}")

# 准备量化数据目录
quant_dir = Path("dataset_quant_npy")
if quant_dir.exists():
    shutil.rmtree(quant_dir)
quant_dir.mkdir(exist_ok=True)

lines = []
for k, i in enumerate(idxs):
    # 取出样本 [4, 256] -> 增加 batch 维 -> [1, 4, 256]
    feat = X[i][np.newaxis, :, :]  
    npy_path = quant_dir / f"{k}.npy"
    np.save(npy_path, feat)
    lines.append(str(npy_path.resolve()))

out_txt = "dataset_quant.txt"
Path(out_txt).write_text("\n".join(lines), encoding="utf-8")
print(f"写入 {len(lines)} 条归一化 npy 路径到 {out_txt}")

rknn = RKNN()
rknn.config(
    target_platform='rk3588',   # 改成你的芯片，如 rk3562 / rk3576
    quantized_dtype='w8a8',
)

print("开始加载 ONNX 模型...")
# 显式指定输入形状 [1, 4, 256]，防止动态形状报错
rknn.load_onnx('model.onnx', inputs=['input'], input_size_list=[[1, 4, 256]])
print("ONNX 模型加载完成")

print("开始构建 RKNN 模型（使用 train.py 归一化后的数据）...")
# rknn-toolkit2 只接受 dataset 为路径字符串，txt 中每行为 npy 绝对路径
rknn.build(
    do_quantization=True,
    dataset=str(Path(out_txt).resolve()),
)
print("RKNN 模型构建完成")

print("开始导出 RKNN 模型...")
rknn.export_rknn('model.rknn')
print("RKNN 模型导出完成")

rknn.release()
