# Week 10: BoVW → MLP in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week10/week10_practice.ipynb)

## Instructions

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Freely modify and run cells in the copied notebook to practice.
4. Complete all Exercises (write code **and run** every cell), then submit the notebook to e-class.

- You may freely modify cells outside of Exercises (only Exercises are graded).
- **Caution**: Redefining variables from earlier cells may break later Exercises.
- The first run downloads the STL-10 dataset (~2.5 GB) into a local cache; subsequent runs reuse it.

## Contents

0. PyTorch Crash Course
1. BoVW Recap & a Reusable Extractor
2. Define an MLP
3. Train the MLP — forward / backward / step
4. Evaluation — Accuracy, Confusion Matrix, P/R/F1
5. Two Quick Comparisons (Linear vs MLP, Sigmoid vs ReLU)
6. Exercise

## Exercises

- [ ] **Exercise 6.1** — Learning rate sweep: train the MLP with `lr ∈ [1e-4, 1e-3, 1e-2, 1e-1, 1.0]` and compare loss curves
