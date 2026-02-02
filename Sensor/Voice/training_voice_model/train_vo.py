import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import wandb
from model_vo import CNN_TCN_MTL

'''
# 需先安装: pip install wandb，并 wandb login
# 睡姿 5 分类（排除最后一类），wandb 监控
python3 train_MLT.py \
    --snore_dir ./dataset_mel \
    --posture_dir ./features_mel \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --save_path ./mlt_best.pth \
    --val_ratio 0.2 \
    --patience 5 \
    --posture_loss_weight 0.5 \
    --wandb_project snore_mtl \
    --wandb_run_name mtl_5class
'''

# 睡姿排除的类别（最后一类），剩余为 5 类
EXCLUDE_POSTURE = 5
NUM_POSTURE_CLASSES = 5


class MergedSnorePostureDataset(Dataset):
    """合并 snore_dir + posture_dir；睡姿排除最后一类，重映射为 0..NUM_POSTURE_CLASSES-1。"""
    def __init__(self, snore_dir, posture_dir, exclude_posture=EXCLUDE_POSTURE, num_posture_classes=NUM_POSTURE_CLASSES):
        self.samples = []

        # -------- snore dataset --------
        for root, _, files in os.walk(snore_dir):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                path = os.path.join(root, f)
                name = f.lower()
                if "_snoring_" in name:
                    snore_label = 1
                elif "_silent_" in name:
                    snore_label = 0
                else:
                    raise ValueError(f"Bad snore filename: {f}")
                self.samples.append({"path": path, "snore": snore_label, "posture": -1})

        # -------- posture dataset（排除 exclude_posture，睡姿重映射为 0..num_posture_classes-1）--------
        posture_raw = []
        for root, _, files in os.walk(posture_dir):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                path = os.path.join(root, f)
                try:
                    p = int(f.split("-")[-1].split(".")[0])
                except Exception:
                    continue
                if p == exclude_posture:
                    continue
                posture_raw.append((path, p))
        unique_p = sorted(set(p for _, p in posture_raw))
        if len(unique_p) != num_posture_classes:
            raise ValueError(f"筛除睡姿 {exclude_posture} 后共有 {len(unique_p)} 类: {unique_p}，与 num_posture_classes={num_posture_classes} 不一致")
        label_map = {old: i for i, old in enumerate(unique_p)}
        for path, p in posture_raw:
            self.samples.append({
                "path": path,
                "snore": 1,
                "posture": label_map[p]
            })

        print(f"Total samples: {len(self.samples)}（睡姿已排除类别 {exclude_posture}，共 {num_posture_classes} 类: {unique_p}）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        feat = np.load(s["path"])
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        return {
            "feat": feat,
            "snore_label": torch.tensor(s["snore"]),
            "posture_label": torch.tensor(s["posture"])
        }



# =====================
# 训练函数
# =====================
def train_model(
    dataset,
    model,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    val_ratio=0.2,
    posture_loss_weight=0.5,
    save_path="./mtl_model.pth",
    patience=5,
    wandb_project="snore_mtl",
    wandb_run_name=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------- Dataset split --------
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # -------- Loss & Optim --------
    snore_criterion = nn.CrossEntropyLoss()
    posture_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- wandb 初始化 --------
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_ratio": val_ratio,
        "posture_loss_weight": posture_loss_weight,
        "patience": patience,
        "snore_classes": 2,
        "posture_classes": NUM_POSTURE_CLASSES,
        "exclude_posture": EXCLUDE_POSTURE,
        "save_path": save_path,
    })

    best_val_loss = float("inf")
    wait = 0

    # =====================
    # Training loop
    # =====================
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        snore_correct, snore_total = 0, 0
        posture_correct, posture_total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch in loop:
            x = batch["feat"].to(device)
            snore_label = batch["snore_label"].to(device)
            posture_label = batch["posture_label"].to(device)

            optimizer.zero_grad()

            snore_logits, posture_logits = model(x)

            # -------- Snore loss (always valid) --------
            snore_loss = snore_criterion(snore_logits, snore_label)

            # -------- Posture loss (masked) --------
            posture_mask = posture_label != -1
            if posture_mask.any():
                posture_loss = posture_criterion(
                    posture_logits[posture_mask],
                    posture_label[posture_mask]
                )
            else:
                posture_loss = torch.tensor(0.0, device=device)

            loss = snore_loss + posture_loss_weight * posture_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # -------- Snore accuracy --------
            snore_pred = snore_logits.argmax(dim=1)
            snore_correct += (snore_pred == snore_label).sum().item()
            snore_total += snore_label.size(0)

            # -------- Posture accuracy (masked) --------
            if posture_mask.any():
                posture_pred = posture_logits.argmax(dim=1)
                posture_correct += (
                    posture_pred[posture_mask] == posture_label[posture_mask]
                ).sum().item()
                posture_total += posture_mask.sum().item()

        # -------- Train metrics --------
        snore_acc = 100. * snore_correct / snore_total
        posture_acc = 100. * posture_correct / max(1, posture_total)
        train_loss_avg = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Snore Acc: {snore_acc:.2f}%, Posture Acc: {posture_acc:.2f}%")
        wandb.log({
            "Train/Loss": train_loss_avg,
            "Train/Snore_Acc": snore_acc,
            "Train/Posture_Acc": posture_acc,
        }, step=epoch)

        # =====================
        # Validation
        # =====================
        model.eval()
        val_posture_loss = 0
        val_posture_count = 0
        val_posture_correct = 0
        val_snore_correct = 0
        val_snore_count = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["feat"].to(device)
                snore_label = batch["snore_label"].to(device)
                posture_label = batch["posture_label"].to(device)

                snore_logits, posture_logits = model(x)

                # Posture loss (masked)，Posture accuracy
                posture_mask = posture_label != -1
                if posture_mask.any():
                    posture_loss = posture_criterion(
                        posture_logits[posture_mask],
                        posture_label[posture_mask]
                    )
                    val_posture_loss += posture_loss.item()
                    val_posture_count += posture_mask.sum().item()
                    posture_pred = posture_logits.argmax(dim=1)
                    val_posture_correct += (posture_pred[posture_mask] == posture_label[posture_mask]).sum().item()
                    
                # -------- Snore accuracy --------
                snore_pred = snore_logits.argmax(dim=1)
                val_snore_correct += (snore_pred == snore_label).sum().item()
                val_snore_count += snore_label.size(0)
                
        avg_val_posture_loss = val_posture_loss / max(1, val_posture_count)
        val_snore_acc = 100. * val_snore_correct / val_snore_count
        val_posture_acc = 100. * val_posture_correct / max(1, val_posture_count)
        print(f"Epoch {epoch+1}/{epochs}, Val Snore Acc: {val_snore_acc:.2f}%, Val Posture Acc: {val_posture_acc:.2f}%")
        wandb.log({
            "Val/Posture_Loss": avg_val_posture_loss,
            "Val/Snore_Acc": val_snore_acc,
            "Val/Posture_Acc": val_posture_acc,
        }, step=epoch)

        # -------- Early stopping --------
        if avg_val_posture_loss < best_val_loss:
            best_val_loss = avg_val_posture_loss
            torch.save(model.state_dict(), save_path)
            wait = 0
            print(f"✅ Model saved to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping")
                break

    wandb.finish()
    print("🎉 Training finished")

def main():
    parser = argparse.ArgumentParser("Snore + Posture Multi-Task Training")

    # -------- Dataset --------
    parser.add_argument("--snore_dir", type=str, required=True,
                        help="snore / non-snore feature directory")
    parser.add_argument("--posture_dir", type=str, required=True,
                        help="snore + posture feature directory")

    # -------- Training --------
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--posture_loss_weight", type=float, default=0.5)

    # -------- Output --------
    parser.add_argument("--save_path", type=str, default="./mtl_best.pth")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--wandb_project", type=str, default="snore_mtl", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name (optional)")

    args = parser.parse_args()
    # =====================
    # Dataset（睡姿排除最后一类，5 分类）
    # =====================
    dataset = MergedSnorePostureDataset(
        snore_dir=args.snore_dir,
        posture_dir=args.posture_dir,
        exclude_posture=EXCLUDE_POSTURE,
        num_posture_classes=NUM_POSTURE_CLASSES,
    )
    print(f"✅ Dataset loaded, total samples: {len(dataset)}")

    # =====================
    # Model（睡姿 5 分类）
    # =====================
    model = CNN_TCN_MTL(snore_classes=2, posture_classes=NUM_POSTURE_CLASSES)
    print("✅ Model initialized (posture_classes=5)")

    # =====================
    # Train
    # =====================
    train_model(
        dataset=dataset,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        posture_loss_weight=args.posture_loss_weight,
        save_path=args.save_path,
        patience=args.patience,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()