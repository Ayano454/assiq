# PyTorch CIFAR-10 Differential Testing Starter

This repository is a **PyTorch starter project** for the assignment on **DeepXplore-style differential testing**.
It trains two CIFAR-10 ResNet50 models and runs a differential-testing script that:

- compares the predictions of two models on the same input,
- saves at least 5 disagreement-inducing inputs,
- reports simple neuron coverage for both models,
- writes outputs to the `results/` directory.

## Important note

The original DeepXplore codebase was released for an older stack. This repository provides a **clean PyTorch pipeline inspired by DeepXplore's core ideas** rather than a line-by-line port of the original implementation. In your report, describe any modifications and clearly state what you implemented.

## Repository structure

```text
.
├── requirements.txt
├── README.md
├── test.py
├── src/
│   ├── data.py
│   ├── coverage.py
│   ├── resnet_cifar.py
│   ├── train.py
│   └── utils.py
├── models/
└── results/
```

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Train model A

```bash
python -m src.train \
  --output ./models/model_a.pth \
  --seed 1 \
  --lr 0.1 \
  --epochs 30
```

## 3) Train model B

```bash
python -m src.train \
  --output ./models/model_b.pth \
  --seed 42 \
  --lr 0.05 \
  --epochs 30 \
  --strong-augment
```

## 4) Run differential testing

```bash
python test.py \
  --checkpoint-a ./models/model_a.pth \
  --checkpoint-b ./models/model_b.pth \
  --output-dir ./results \
  --num-visualizations 5
```

## Outputs

After running `test.py`, the `results/` directory should contain:

- `disagreement_01.png`, `disagreement_02.png`, ...
- `summary.json`

The summary file includes:

- number of test inputs,
- number of disagreements,
- disagreement rate,
- number of saved visualizations,
- simple neuron coverage for each model.

## Suggested discussion points for the report

- What kinds of CIFAR-10 images caused disagreement?
- Were they visually ambiguous, low-contrast, or semantically similar classes?
- Did one model tend to be more robust than the other?
- How did coverage compare across the two models?

## Files required by the assignment

Add or keep the following in your GitHub repository:

- `requirements.txt`
- `test.py`
- `results/`
- `report.pdf`
- `README.md`

## Suggested commit flow

Instead of one large commit, use incremental commits such as:

1. `init project structure`
2. `add cifar10 resnet50 model`
3. `add training script`
4. `add differential testing script`
5. `save result visualizations`
6. `write README and finalize report`
