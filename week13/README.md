# Week 13: Standing on Giants — Running Pretrained Vision Models from the Hugging Face Hub

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week13/week13_practice.ipynb)

This is the **final lab**, so there are no graded exercises — we run everything together in
class. To follow along on your own machine:

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Run the cells top to bottom, then tweak them freely — swap models, labels, thresholds,
   and images to see what changes.

- The first run **downloads model weights** and caches them (YOLOS ~130 MB up to CLIP ~600 MB,
  BLIP ~990 MB). Everything runs on **CPU** — no GPU needed. One slow spot: SAM's
  segment-everything cell (Section 4) takes about a minute on CPU; every other model,
  including BLIP captioning, runs in a second or two.

## Contents

0. The Hugging Face Hub & `pipeline` — 3 lines to a prediction (ViT classification)
1. Same 3 lines, a different model — swap checkpoints (ResNet, ConvNeXt)
2. Zero-shot classification with CLIP — labels you invent
3. Object detection — YOLO (YOLOS)
4. Segment everything — SAM (`mask-generation`)
5. Bonus: image captioning — BLIP (processor → model → decode)

## Hands-on (we do these together)

- **After Section 1** — run another classifier from the Hub (Swin Transformer) and compare its top-1 with ViT's
- **After Section 2** — zero-shot the street scene with our own labels + a prompt-template test
- **After Section 3** — run the detector on a new street image, draw boxes, and count one class
