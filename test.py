from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.coverage import NeuronCoverageTracker
from src.data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, build_test_loader
from src.resnet_cifar import build_model
from src.utils import ensure_dir, save_json



def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)



def save_visualization(
    image: torch.Tensor,
    label: int,
    pred_a: int,
    pred_b: int,
    output_path: str,
) -> None:
    img = denormalize(image).permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"true: {CLASS_NAMES[label]}\n"
        f"model_a: {CLASS_NAMES[pred_a]} | model_b: {CLASS_NAMES[pred_b]}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()



def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepXplore-style differential testing on two CIFAR-10 ResNet50 models."
    )
    parser.add_argument("--checkpoint-a", type=str, required=True)
    parser.add_argument("--checkpoint-b", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument("--coverage-threshold", type=float, default=0.2)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = build_test_loader(args.data_dir, args.batch_size, args.num_workers)

    model_a = load_model(args.checkpoint_a, device)
    model_b = load_model(args.checkpoint_b, device)

    tracker_a = NeuronCoverageTracker(threshold=args.coverage_threshold)
    tracker_b = NeuronCoverageTracker(threshold=args.coverage_threshold)
    tracker_a.register(model_a)
    tracker_b.register(model_b)

    total = 0
    disagreements = 0
    saved = 0
    both_wrong = 0
    one_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits_a = model_a(images)
            logits_b = model_b(images)
            preds_a = logits_a.argmax(dim=1)
            preds_b = logits_b.argmax(dim=1)

            disagree_mask = preds_a != preds_b
            total += images.size(0)
            disagreements += disagree_mask.sum().item()

            correct_a = preds_a == labels
            correct_b = preds_b == labels
            one_correct += ((correct_a ^ correct_b) & disagree_mask).sum().item()
            both_wrong += ((~correct_a) & (~correct_b) & disagree_mask).sum().item()

            disagree_indices = torch.where(disagree_mask)[0].tolist()
            for idx in disagree_indices:
                if saved >= args.num_visualizations:
                    break
                filename = os.path.join(args.output_dir, f"disagreement_{saved + 1:02d}.png")
                save_visualization(
                    image=images[idx],
                    label=int(labels[idx].item()),
                    pred_a=int(preds_a[idx].item()),
                    pred_b=int(preds_b[idx].item()),
                    output_path=filename,
                )
                saved += 1

    summary = {
        "num_test_inputs": total,
        "num_disagreements": disagreements,
        "disagreement_rate": disagreements / total if total else 0.0,
        "num_visualizations_saved": saved,
        "disagreement_one_correct": one_correct,
        "disagreement_both_wrong": both_wrong,
        "model_a_coverage": tracker_a.summary(),
        "model_b_coverage": tracker_b.summary(),
        "notes": [
            "This is a PyTorch differential-testing pipeline inspired by DeepXplore.",
            "It reports prediction disagreements and simple neuron coverage over ReLU activations.",
        ],
    }

    save_json(os.path.join(args.output_dir, "summary.json"), summary)

    print("=== Differential Testing Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    tracker_a.close()
    tracker_b.close()


if __name__ == "__main__":
    main()
