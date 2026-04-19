# Week 7: Classic Features & Matching

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeongminLEE/image-processing-tutorial/blob/main/week7/week7_practice.ipynb)

## Instructions

1. Click the **Open In Colab** button above.
2. Go to **File > Save a copy in Drive** to save it to your personal Google Drive.
3. Freely modify and run cells in the copied notebook to practice.
4. Complete all Exercises (write code **and run** every cell), then submit the notebook to e-class.

- You may freely modify cells outside of Exercises (only Exercises are graded).
- **Caution**: Redefining variables from earlier cells may break later Exercises.

## Contents

0. Why Features?
1. Harris Corner Detection
2. SIFT — Keypoints & Descriptors
3. Descriptor Space
4. Feature Matching with Lowe's Ratio Test
5. RANSAC — Robust Homography Estimation

## Exercises

- [ ] **Exercise 1.1** — Sweep the Harris response threshold (0.001 / 0.01 / 0.1) and compare corner overlays
- [ ] **Exercise 3.1** — Find the nearest-neighbor descriptor by hand with `np.linalg.norm`, report `d_min` and `d_second`
- [ ] **Exercise 4.1** — Sweep Lowe's ratio threshold (0.5 / 0.7 / 0.9) and observe the precision–recall trade-off
- [ ] **Exercise 5.1** — Compare `findHomography` with and without RANSAC by warping `img1` onto `img2`'s frame
