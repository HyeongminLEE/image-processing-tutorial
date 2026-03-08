# Week 4 Final Challenge: Bilateral Filter Implementation

**Source:** `week4/week4_practice.ipynb` — Final Challenge section
**Week topic:** Noise & Edge Detection

## Context

Week 4 covers noise models (Gaussian, salt-and-pepper), denoising filters
(mean, Gaussian, median), quality metrics (PSNR, SSIM), and edge detection
(Sobel, Laplacian, Canny). Students learn that standard smoothing filters
reduce noise but also blur edges.

The **bilateral filter** is introduced as a solution: it uses both spatial
proximity and intensity similarity to preserve edges while smoothing.

## Challenge Description

### Bilateral Filter: Edge-Preserving Denoising

All the filters used in the week (mean, Gaussian, median) share a common
limitation: they smooth **everything**, including edges.

The **bilateral filter** solves this by using two Gaussian weights:
- **Spatial weight** $G_{\sigma_s}(\|p - q\|)$ — nearby pixels contribute more
- **Range weight** $G_{\sigma_r}(|I_p - I_q|)$ — pixels with **similar intensity** contribute more

$$\text{BF}[I]_p = \frac{1}{W_p}\sum_{q \in N(p)} G_{\sigma_s}(\|p-q\|) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q$$

where $W_p = \sum_q G_{\sigma_s} \cdot G_{\sigma_r}$ is a normalization factor.

**Key insight**: At an edge, pixels on the opposite side have very different
intensity, so their range weight becomes near-zero. The filter effectively
**stops at edges** while still averaging similar pixels.

A reference figure from Durand & Dorsey (2002) illustrates the bilateral
filtering process for a single pixel (spatial kernel + range kernel → combined).

**Task:** Complete the `bilateral_filter` function. The outer loop, padding,
and spatial weight precomputation are provided. Students fill in **4 lines**:

1. Compute **range weights**: Gaussian of intensity difference
2. Combine spatial and range weights
3. Normalize so weights sum to 1
4. Compute the weighted average

Then apply to a noisy image and compare with standard Gaussian using PSNR.

**Hint:** Range weight for each neighbor is
$\exp\!\left(-\frac{(I_\text{neighbor} - I_\text{center})^2}{2\sigma_r^2}\right)$.
Compute for the entire neighborhood at once using NumPy broadcasting.

## Scaffolding Code

```python
def bilateral_filter(img, d=5, sigma_s=2.0, sigma_r=25.0):
    """Bilateral filter: edge-preserving denoising.

    Parameters
    ----------
    img     : 2D uint8 array (grayscale image)
    d       : kernel diameter (must be odd)
    sigma_s : spatial Gaussian standard deviation
    sigma_r : range (intensity) Gaussian standard deviation
    """
    img_f = img.astype(np.float64)
    h, w = img_f.shape
    r = d // 2
    output = np.zeros_like(img_f)

    # Pre-compute spatial Gaussian weights (same for every pixel)
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    spatial_weights = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2))

    # Pad image for border handling
    padded = np.pad(img_f, r, mode="reflect")

    for i in range(h):
        for j in range(w):
            # Extract the local neighborhood
            neighborhood = padded[i : i + d, j : j + d]
            center_val = img_f[i, j]

            # YOUR CODE HERE
            # 1. range_weights = Gaussian of (neighborhood - center_val)
            # 2. combined_weights = spatial_weights * range_weights
            # 3. Normalize combined_weights so they sum to 1
            # 4. output[i, j] = weighted sum of neighborhood

    return np.clip(output, 0, 255).astype(np.uint8)
```

```python
# Test
np.random.seed(42)
noisy = add_gaussian_noise(img_gray, 25)
show_images(img_gray, noisy, titles=["Original (img_gray)", "Noisy (σ=25)"])

# YOUR CODE HERE
# 1. bf_result = bilateral_filter(noisy, d=5, sigma_s=2.0, sigma_r=25.0)
# 2. gauss_result — apply gaussian_filter(noisy, sigma=1.0) for comparison

show_denoising_comparison(img_gray, [noisy, gauss_result, bf_result],
                          ["Noisy", "Gaussian filter", "Bilateral filter"])
```

## Why This Works as a Project Problem

- Implementation challenge: students must understand the math to fill in 4 lines
- Directly contrasts with standard filters covered in the week
- PSNR comparison gives quantitative evidence of edge preservation
- The bilateral filter is widely used in practice (Photoshop, OpenCV)
- Figure reference: Durand & Dorsey, 2002

## Resources

- `add_gaussian_noise()`: helper defined in the notebook
- `show_denoising_comparison()`: helper for side-by-side PSNR display
- Reference figure: `week4/bilateral_filter_fig6.png` (GitHub raw URL in notebook)
