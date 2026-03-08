# Week 3 Final Challenge: Hybrid Image Separation

**Source:** `week3/week3_practice.ipynb` — Final Challenge section
**Week topic:** Filtering & Frequency Domain — Convolution, 2D FFT

## Context

Week 3 covers spatial filtering (mean, Gaussian, sharpening) and frequency-domain
analysis (2D FFT, low-pass/high-pass filtering). Students learn that low-pass
filtering extracts smooth structure while high-pass filtering extracts edges/detail.

The **Einstein–Monroe hybrid image** is shown during the lecture: it encodes
Einstein in high frequencies and Monroe in low frequencies. Up close you see
Einstein; from far away (or when blurred) Monroe appears.

## Challenge Description

**Task:** Separate the two faces from the hybrid image.

1. Load the hybrid image (`CHALLENGE_PATH`) as grayscale
2. Apply a low-pass filter to extract one face — any method from the week:
   Gaussian blur, frequency domain masking, or anything else
3. Subtract the low-pass result from the original to get the other face
   (the two components should sum to the original)
4. Display three images: the hybrid, the low-frequency face, and the
   high-frequency face
5. Tune filter parameters by **checking results visually** until clean separation

**Hint:** The high-frequency result will have negative values — use
`rescale_for_display` to visualize it.

## Scaffolding Code

```python
img_challenge = np.array(Image.open(CHALLENGE_PATH).convert("L"))
print(f"Hybrid image — shape: {img_challenge.shape}, dtype: {img_challenge.dtype}")
show_images(img_challenge, titles=["Einstein–Monroe hybrid image"])
```

```python
# YOUR CODE HERE
# Separate the two faces from img_challenge:
# 1. Apply a low-pass filter → low_freq
# 2. Subtract: high_freq = img_challenge.astype(np.float64) - low_freq

show_images(img_challenge,
            np.clip(low_freq, 0, 255).astype(np.uint8),
            rescale_for_display(high_freq),
            titles=["Hybrid", "Low-frequency face", "High-frequency face"])
```

## Why This Works as a Project Problem

- Tests understanding of frequency decomposition (core concept of week 3)
- Open-ended: students choose their own filtering approach and tune parameters
- Visual verification: students can immediately see if their separation is clean
- Connects theory (frequency domain) to a memorable real-world artifact

## Resources

- Hybrid image file: loaded via `CHALLENGE_PATH` in the notebook's Colab setup
- `rescale_for_display()`: helper defined in the notebook that maps float arrays
  to [0, 255] uint8 for visualization
