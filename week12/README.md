# Week 12: Transformers — Self-Attention, and Building One in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week12/week12_practice.ipynb)

## Instructions

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Freely modify and run cells in the copied notebook to practice.
4. Complete all Exercises (write code **and run** every cell), then submit the notebook to e-class.

- You may freely modify cells outside of Exercises (only Exercises are graded).
- **Caution**: Redefining variables from earlier cells may break later Exercises.

## Contents

1. Self-Attention from Scratch — Q/K/V, scaled dot-product, softmax, weighted sum
2. The Same Thing: `nn.MultiheadAttention` — input/output shapes, multi-head
3. Stacking Blocks + Positional Encoding — `nn.TransformerEncoder`, permutation invariance
4. Train a Tiny Transformer to *Reverse* a Sequence — and read its attention map
5. Bonus: Pretrained Transformers in 3 Lines (Hugging Face)

## Exercises

- [ ] **Exercise 2.1** — Predict the output and attention-weight shapes for a multi-head attention call
- [ ] **Exercise 4.1** — Train the model on a different task (copy / shift) and predict its attention map
- [ ] **Exercise 4.2** — Declare your own encoder with `nn.TransformerEncoder` and train it on the reverse task
