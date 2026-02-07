"""
使用 data/ 下的数据训练睡姿 5 分类模型，类别由文件名「最后一级」解析（支持二级、三级命名）。
保存为 .pth。
python train.py --data_dir data --epochs 80 --batch_size 32 --lr 0.001 --save_path posture_5class.pth
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model_pr import PostureClassifier, NUM_CLASSES, CLASS_NAMES


class FocalLoss(nn.Module):
    """Focal loss：对难样本加权，gamma=0 退化为 CE。"""

    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.gamma <= 0:
            return nn.functional.cross_entropy(
                logits, targets, weight=self.weight, label_smoothing=self.label_smoothing
            )
        log_p = nn.functional.log_softmax(logits, dim=1)
        pt = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma * log_p.gather(1, targets.unsqueeze(1)).squeeze(1))
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def parse_class_from_filename(filepath):
    """
    从文件名解析类别：最后一级的数字。
    例如 001-00-2.xlsx -> 2,  01-0.csv -> 0,  001-01-1.csv -> 1。
    仅接受 0~4（5 类），否则返回 -1。
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    parts = base.split("-")
    if not parts:
        return -1
    last = parts[-1].strip()
    try:
        label = int(last)
    except ValueError:
        return -1
    if 0 <= label <= 4:
        return label
    return -1


# 支持的数据文件后缀
DATA_EXTENSIONS = (".csv", ".xlsx", ".xls")


def load_file(filepath):
    """
    读单个 csv / xlsx，去掉首列时间戳，若首行为 0-255 表头则去掉。
    切分规则：
    - 行数 > 6：拆成两组，取 first4 和 last4，返回两个 (4,256) 样本；
    - 行数 <= 6：取中间 4 行，返回一个 (4,256) 样本。
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in DATA_EXTENSIONS:
        return []
    if ext == ".csv":
        try:
            df = pd.read_csv(filepath, header=None, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(filepath, header=None, encoding="utf-8-sig")
    else:
        # .xlsx / .xls
        df = pd.read_excel(filepath, header=None)
    df = df.iloc[:, 1:]  # 去掉时间戳列
    arr = pd.to_numeric(df.values.ravel(), errors="coerce").reshape(df.shape).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # CSV 可能首行是 0,1,2,...,255 表头
    if arr.shape[0] > 4 and ext == ".csv":
        first_row = arr[0]
        if np.all(np.isfinite(first_row)) and first_row[0] <= 1 and first_row[-1] >= 250:
            arr = arr[1:]
    n = arr.shape[0]
    if n < 4:
        return []
    if n > 6:
        return [arr[:4], arr[-4:]]
    # n 在 4~6：取中间 4 行
    start = (n - 4) // 2
    return [arr[start : start + 4]]


def load_all_data(data_dir="data"):
    """从 data_dir 下所有 .csv、.xlsx（及 .xls）加载，类别取文件名最后一级。"""
    data_dir = os.path.abspath(data_dir)
    files = []
    for ext in (".csv", ".xlsx", ".xls"):
        files.extend(glob.glob(os.path.join(data_dir, "*" + ext)))
    all_X, all_Y = [], []
    for filepath in files:
        try:
            samples = load_file(filepath)
            if not samples:
                continue
            class_label = parse_class_from_filename(filepath)
            if class_label < 0:
                continue
            for arr in samples:
                if arr.shape != (4, 256):
                    continue
                all_X.append(arr)
                all_Y.append(class_label)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue
    if not all_X:
        raise ValueError(f"No valid samples in {data_dir}")
    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    x_max = X.max()
    if x_max <= 0:
        x_max = 1.0
    X = X / x_max
    return X, Y


def main():
    parser = argparse.ArgumentParser(description="Train posture 5-class classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory (csv/xlsx)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--save_path", type=str, default="posture_5class.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 regularization")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping: stop if test acc no improvement for N epochs")
    parser.add_argument("--label_smoothing", type=float, default=0, help="CrossEntropy label smoothing (0 to disable)")
    parser.add_argument("--class_weight", action="store_true", help="Use balanced class weight for imbalanced data")
    parser.add_argument("--scheduler", action="store_true", default=True, help="Use ReduceLROnPlateau when test acc plateaus")
    parser.add_argument("--aug_noise", type=float, default=0.01, help="Train augmentation: Gaussian noise std (0=off)")
    parser.add_argument("--aug_scale", type=float, default=0.08, help="Train augmentation: scale jitter range (0=off)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha (0=off, try 0.2 to boost)")
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Focal loss gamma (0=CE, try 1.0~2.0 for hard examples)")
    parser.add_argument("--n_runs", type=int, default=1, help="Train n times with different seeds, save best each (for ensemble)")
    args = parser.parse_args()

    for run in range(args.n_runs):
        seed = args.seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        save_path_run = args.save_path
        if args.n_runs > 1:
            base, ext = os.path.splitext(args.save_path)
            save_path_run = f"{base}_run{run+1}{ext}"
            print(f"\n{'='*50} Run {run+1}/{args.n_runs} (seed={seed}) {'='*50}")

        print("Loading data...")
        X, Y = load_all_data(args.data_dir)
        print(f"Total samples: {X.shape[0]}, shape: {X.shape}")
        for c in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[c]}: {np.sum(Y == c)}")

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=args.test_ratio, random_state=seed, stratify=Y
        )

        class_weight = None
        if args.class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(Y_train)
            w = compute_class_weight("balanced", classes=classes, y=Y_train)
            class_weight = torch.tensor(w, dtype=torch.float32)
            print(f"Class weights: {dict(zip(classes, w.round(3)))}")

        train_ds = TensorDataset(
            torch.tensor(X_train), torch.tensor(Y_train, dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test), torch.tensor(Y_test, dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if run == 0:
            print(f"Using device: {device}")
            if device.type == "cuda":
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"aug_noise={args.aug_noise}  aug_scale={args.aug_scale}  mixup={args.mixup_alpha}  focal_gamma={args.focal_gamma}")
        model = PostureClassifier().to(device)
        if class_weight is not None:
            class_weight = class_weight.to(device)
        if args.focal_gamma > 0:
            criterion = FocalLoss(weight=class_weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
            )

        best_test_acc = 0.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                # 数据增强（仅训练时）
                if model.training and (args.aug_noise > 0 or args.aug_scale > 0):
                    if args.aug_noise > 0:
                        bx = bx + args.aug_noise * torch.randn_like(bx, device=bx.device)
                    if args.aug_scale > 0:
                        scale = 1.0 + (2 * torch.rand(1, device=bx.device).item() - 1) * args.aug_scale
                        bx = bx * scale
                # Mixup（仅训练时）
                if model.training and args.mixup_alpha > 0:
                    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                    perm = torch.randperm(bx.size(0), device=bx.device)
                    bx = lam * bx + (1 - lam) * bx[perm]
                    by_a, by_b = by, by[perm]
                optimizer.zero_grad()
                logits = model(bx)
                if model.training and args.mixup_alpha > 0:
                    log_p = torch.log_softmax(logits, dim=1)
                    loss = -(lam * log_p.gather(1, by_a.unsqueeze(1)).squeeze(1) + (1 - lam) * log_p.gather(1, by_b.unsqueeze(1)).squeeze(1)).mean()
                else:
                    loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)
            train_acc = 100.0 * correct / total

            model.eval()
            t_correct, t_total = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx).argmax(dim=1)
                    t_correct += (pred == by).sum().item()
                    t_total += by.size(0)
            test_acc = 100.0 * t_correct / t_total

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path_run)
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  "
                      f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%  best_test={best_test_acc:.2f}% (ep{best_epoch})")

            if scheduler is not None:
                scheduler.step(test_acc)

            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

        print(f"Best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}, model saved to {save_path_run}")
        if args.n_runs > 1:
            print(f"Ensemble: load all *_run*.pth, average logits then argmax for prediction.")


if __name__ == "__main__":
    main()
