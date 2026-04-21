from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from src.data import build_dataloaders
from src.resnet_cifar import build_model
from src.utils import accuracy, ensure_dir, seed_everything



def evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_acc += accuracy(logits, labels) * images.size(0)

    size = len(loader.dataset)
    return total_loss / size, total_acc / size



def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 ResNet50 model.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./models/model_a.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--strong-augment", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output) or ".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        strong_augment=args.strong_augment,
    )

    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = MultiStepLR(optimizer, milestones=[15, 22, 27], gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(logits.detach(), labels)
            running_loss += loss.item() * images.size(0)
            running_acc += batch_acc * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_acc / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "seed": args.seed,
                    "strong_augment": args.strong_augment,
                    "lr": args.lr,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Final test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
