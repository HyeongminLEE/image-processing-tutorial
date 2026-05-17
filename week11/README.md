# Week 11: CNN — Conv2d, Pooling, and Training a Small CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week11/week11_practice.ipynb)

## Instructions

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Freely modify and run cells in the copied notebook to practice.
4. Complete all Exercises (write code **and run** every cell), then submit the notebook to e-class.

- You may freely modify cells outside of Exercises (only Exercises are graded).
- **Caution**: Redefining variables from earlier cells may break later Exercises.
- The first run downloads STL-10 (~2.5 GB) and a pretrained ResNet-18 (~45 MB).

## Contents

0. STL-10 — Same 5-Class Subset as Week 10
1. `nn.Conv2d` — A Learnable Filter (parameter count, output size, translation equivariance)
2. `nn.MaxPool2d` — Downsample Without Parameters
3. Build & Train a Small CNN (CNN vs flattened-pixel MLP)
4. Data Augmentation with `torchvision.transforms.v2`
5. Grad-CAM — Where Is the CNN Looking?

## Exercises

- [ ] **Exercise 3.1** — Tweak the CNN architecture (add a block, widen channels, or change kernel size) and compare to the baseline
