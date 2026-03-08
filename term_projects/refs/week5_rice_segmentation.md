# Week 5 Final Challenge: Rice Grain Segmentation

**Source:** `week5/week5_practice.ipynb` — Final Challenge section
**Week topic:** Segmentation & Morphology — Color Spaces, Thresholding, Morphological Ops

## Context

Week 5 covers the full segmentation pipeline: color space conversion,
thresholding (global/Otsu/adaptive), morphological operations (erosion,
dilation, opening, closing), connected component labeling, and region
property analysis. Students practice the pipeline on a coins image.

The **rice grain** image poses a harder challenge: many small objects with
**non-uniform illumination** (brighter center, darker edges), making global
thresholding fail.

## Challenge Description

### Rice Grain Segmentation

The rice image has many small grains on a dark background with non-uniform
illumination. **Task: count the rice grains and measure their sizes.**

Pipeline steps:

1. **Load** the image and examine its histogram — is it bimodal?
2. **Threshold**: choose between Otsu and adaptive. Which works better
   given the illumination variation?
3. **Morphology**: apply opening + closing. Experiment with iterations —
   rice grains are much smaller than coins, so parameters differ.
4. **Label**: use connected components to label individual grains.
5. **Filter**: remove very small regions (noise) by setting an area threshold.
6. **Report**: print grain count and area statistics (mean, min, max).

Check results **visually** at each step and adjust parameters. Use
`overlay_contours` to display the final result.

**Hint:** Start by examining the histogram to decide between Otsu and
adaptive. The rice image has illumination variation, so think about which
method handles that better. For morphology, start with 1 iteration and
increase if needed — rice grains are small so aggressive morphology may
remove them.

## Scaffolding Code

```python
# Step 1: Load and examine
rice = np.array(Image.open(RICE_PATH))
print(f"Rice image — shape: {rice.shape}, dtype: {rice.dtype}")

show_with_hist(rice, titles=["Rice image"])

# YOUR CODE HERE
# Step 2: Threshold (choose Otsu or adaptive)
# Step 3: Morphology (opening + closing, tune iterations)
# Step 4: Label (connected components)
# Step 5: Filter by area
# Store final results in: rice_mask (bool), rice_labels (int), n_rice (int)


# Step 6: Report results
areas = [r.area for r in regionprops(rice_labels)]
print(f"\nRice grains detected: {n_rice}")
print(f"Area statistics:")
print(f"  Mean: {np.mean(areas):.1f} pixels")
print(f"  Min:  {np.min(areas)} pixels")
print(f"  Max:  {np.max(areas)} pixels")

# Overlay on original
rice_overlay = overlay_contours(rice, rice_mask)
show_images(rice_overlay, titles=[f"Result: {n_rice} rice grains"])
show_labeled(rice_labels, title=f"Labeled rice grains")
```

## Why This Works as a Project Problem

- End-to-end pipeline: tests all week 5 concepts in sequence
- Non-uniform illumination forces students to choose adaptive over Otsu
- Parameter tuning (morphology iterations, area threshold) requires reasoning
- Different from the coins example — students must adapt, not copy
- Quantitative output (grain count, area stats) makes grading objective

## Resources

- Rice image: downloaded at runtime via Colab setup cell
  (`https://raw.githubusercontent.com/.../images/rice.png`)
- `overlay_contours()`: helper that draws contours on the original image
- `show_labeled()`: helper that displays labeled regions with distinct colors
- `regionprops()`: from `skimage.measure`, used for area statistics
