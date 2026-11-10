import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders(data_dir, batch_size):
    norm = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(*norm),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(*norm)])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=train_tf)
    val_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, num_classes=10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        preds_all.append(pred.cpu())
        labels_all.append(y.cpu())
    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    cm = torch.bincount(labels * num_classes + preds,
                        minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return loss_sum / total, correct / total, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("reports", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader = get_loaders(args.data, args.batch_size)
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses = [], []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss, val_acc, cm = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, "best.pth")
            print(f"  -> saved best.pth (acc={val_acc:.4f})")

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="train")
    plt.plot(range(1, args.epochs + 1), val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)
    plt.savefig("reports/loss_curve.png", dpi=150, bbox_inches="tight")
    print("saved reports/loss_curve.png")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm.numpy(), cmap="Blues")
    plt.colorbar(); plt.xlabel("predicted"); plt.ylabel("true")
    plt.title(f"confusion matrix (best val_acc={best_acc:.4f})")
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("saved reports/confusion_matrix.png")

    print(f"best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()