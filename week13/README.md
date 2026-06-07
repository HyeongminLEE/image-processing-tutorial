# Week 13: Standing on Giants — Running Pretrained Vision Models from the Hugging Face Hub

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week13/week13_practice.ipynb)

## Instructions

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Freely modify and run cells in the copied notebook to practice.
4. Complete all Exercises (write code **and run** every cell), then submit the notebook to e-class.

- You may freely modify cells outside of Exercises (only Exercises are graded).
- **Caution**: Redefining variables from earlier cells may break later Exercises.
- The first run **downloads model weights** and caches them (YOLOS ~130 MB up to CLIP ~600 MB,
  BLIP ~990 MB). Everything runs on **CPU** — no GPU needed. One slow spot: SAM's
  segment-everything cell (Section 4) takes about a minute on CPU.

## Contents

0. The Hugging Face Hub & `pipeline` — 3 lines to a prediction (ViT classification)
1. Same 3 lines, a different model — swap checkpoints (ResNet, ConvNeXt)
2. Zero-shot classification with CLIP — labels you invent
3. Object detection — YOLO (YOLOS)
4. Segment everything — SAM (`mask-generation`)
5. Bonus: image captioning — BLIP (processor → model → decode)

## Exercises

- [ ] **Exercise 1.1** — Pick any image-classification model from the Hub and run it on the parrots image
- [ ] **Exercise 2.1** — Zero-shot classify the street scene with your own labels + a prompt-template test
- [ ] **Exercise 3.1** — Run the detector on a new image, filter by score, and count one class
