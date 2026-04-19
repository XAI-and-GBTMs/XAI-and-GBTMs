# Interpreting CNN Hierarchies with SHAP

A study on **why high accuracy doesn't mean a model is actually learning what you think it is.**

We built a ResNet18 classifier for pneumonia detection from chest X-rays and got strong numbers on the original dataset. Then we used SHAP to look at what the model actually learned and found it was picking up on the wrong indicators. Cross-dataset evaluation confirmed it: accuracy dropped from 92.8% to 78.0% on a different chest X-ray dataset, which is a significant enough gap to say the model isn't generalizing.

The point of this project isn't the classifier. It's that SHAP caught something the metrics didn't.

---

## What happened

The model trained on the [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) achieved:

- **AUC: 0.9694** on its own test split
- **92.8% accuracy** on in-distribution data (624 images)

Good numbers. But SHAP overlays showed the model wasn't consistently highlighting the lung regions you'd expect — the consolidations, infiltrates, and opacifications that a radiologist looks for. Instead, it was latching onto patterns that happen to correlate with the label in this specific dataset (image borders, scanner-specific noise).

When we ran the model on a separate chest X-ray dataset (Epic Hospital, Chittagong, Bangladesh — 513 images, balanced classes), accuracy dropped to **78.0%** and AUC fell from 0.9694 to 0.7189. That's a ~15 point accuracy gap and a massive AUC drop, consistent with a model that memorized dataset-specific shortcuts rather than clinical pathology.

The model is right for the wrong reasons.

---

## Why this matters

A model that hits 92.8% in a controlled benchmark but drops to 78.0% (AUC 0.72) on a different hospital's scanner is not a deployable diagnostic tool. SHAP made this visible before deployment. Without it, the benchmark numbers alone would have looked like a success.

---

## How it works

### The model

ResNet18 pretrained on ImageNet, with the early layers frozen and only the deeper layers fine-tuned on chest X-rays. The final layer is replaced with a dropout + single linear unit — it outputs one number, and sigmoid turns that into a probability (0 = Normal, 1 = Pneumonia). Training used Binary Cross-Entropy loss with a WeightedRandomSampler to handle the 3:1 class imbalance in the dataset.

### Youden's J threshold

The model outputs a probability, not a hard label. You need a cutoff to decide: above this → pneumonia, below this → normal. The default 0.5 is arbitrary and often wrong, especially when classes are imbalanced.

Youden's J finds the threshold that maximizes `Sensitivity + Specificity - 1`, which geometrically is the point on the ROC curve farthest from the diagonal (the random-guess baseline). It's the threshold where the model is best at both catching real pneumonia cases and not wrongly flagging healthy patients at the same time.

In this project it landed at 0.9886 — much higher than 0.5. That's because the class imbalance during training pushed the model's raw probabilities high, so a 0.5 cutoff would flag nearly everything as pneumonia. Youden's J corrects for this automatically.

### How SHAP works here

SHAP (SHapley Additive exPlanations) answers the question: which pixels in this image pushed the model toward its prediction, and by how much?

It works by masking out regions of the image — here using a strong blur to "hide" patches — and measuring how much the model's output probability drops. Regions that cause a big drop when hidden are the ones the model was relying on. This process is run recursively across the image in a hierarchical partition (hence `PartitionExplainer`), which makes it efficient enough to run on full-resolution images.

The output is a map of attribution scores, one per pixel. Positive values push toward Pneumonia. We take the top percentile of those values, render them as a red intensity overlay, and draw contours around the hottest regions so you can see exactly where the model was "looking."

The key finding: on the Kaggle dataset, the model's SHAP maps were not consistently landing on the lung parenchyma where pneumonia actually appears.

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision shap opencv-python scikit-learn matplotlib numpy
```

### 2. Download the datasets

Download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle (used for training/in-distribution evaluation):
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download the **Chest X-Ray Epic Hospital, Chittagong, Bangladesh** dataset from Mendeley (used for cross-dataset evaluation):
https://data.mendeley.com/datasets/wndbd5r26y/2

Extract them so the directory structure looks like this:

```
CNNs_on_Chest_X_Ray/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia/
│   ├── Training/
│   │   ├── normal/
│   │   └── pneumonia/
│   └── Testing/
│       ├── normal/
│       └── pneumonia/
├── pneumonia_classification.ipynb
└── cross_dataset_evaluation.ipynb
```

### 3. Run the notebook

Pre-trained weights are at `results/best_model.pth`. Just run all cells.

> The notebook auto-detects MPS (Apple Silicon), CUDA, or CPU.

---

## Notebook Overview

| Section | What it does |
|---|---|
| 1. Data Preparation | Augmentation, normalization, ImageFolder loaders |
| 2. Class Imbalance | WeightedRandomSampler to fix 3:1 PNEUMONIA/NORMAL skew |
| 3. Architecture | ResNet18 with frozen early layers + dropout head |
| 4. Training | BCEWithLogitsLoss + Adam, loads best checkpoint |
| 5. Evaluation | ROC-AUC, confusion matrix, Youden's J threshold |
| 6. SHAP | PartitionExplainer with blur masking — where the model actually looks |
