# Week 13: Vision Transformers & CLIP — Using Pretrained Foundation Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week13/week13_practice.ipynb)

## Instructions

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Freely modify and run cells in the copied notebook to practice.
4. Complete all Exercises (write code **and run** every cell), then submit the notebook to e-class.

- You may freely modify cells outside of Exercises (only Exercises are graded).
- **Caution**: Redefining variables from earlier cells may break later Exercises.
- The first run downloads model weights (ViT ~330 MB, CLIP ~600 MB) and caches them; everything runs on CPU.

## Contents

0. The Hugging Face Hub & `pipeline` — 3 lines to an image prediction
1. Opening the Box — Processor + Model (ViT): `pixel_values`, `logits`, `id2label`
2. ViT Sees With Attention — read the CLS→patch attention and overlay it on the image
3. CLIP — Zero-Shot Classification with labels you invent
4. CLIP Embeddings & Text→Image Retrieval — `get_image_features` / `get_text_features`

## Exercises

- [ ] **Exercise 1.1** — Run the manual ViT pipeline on a street scene; why object-centric labels struggle
- [ ] **Exercise 3.1** — Zero-shot classify with your own labels + a prompt-engineering test
