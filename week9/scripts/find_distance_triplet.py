"""Find the most dramatic BoVW distance triplet (same-class pair, far different-class).

For each training image i, find:
    j = nearest same-class image (smallest BoVW L2 distance)
    k = farthest different-class image (largest BoVW L2 distance)
Pick the triplet that maximizes (D[i,k] - D[i,j]) AND has both images with
enough descriptors (so we don't pick outliers).

Saves a 3-panel preview PNG so the user can sanity-check the picks.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.datasets as datasets
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import squareform, pdist

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path.home() / ".cache" / "stl10"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SELECTED = [0, 1, 2, 3, 8]
CLASS_NAMES = ["airplane", "bird", "car", "cat", "ship"]
PER_CLASS_TRAIN = 100
N_DESC_CAP = 100
K_VOCAB = 200


def select_subset(ds, n_per_class):
    new_label = {old: i for i, old in enumerate(SELECTED)}
    items = []
    counts = {c: 0 for c in SELECTED}
    for i in range(len(ds)):
        img, label = ds[i]
        if label not in new_label:
            continue
        if counts[label] >= n_per_class:
            continue
        items.append((np.array(img.convert("L")), new_label[label]))
        counts[label] += 1
        if all(v >= n_per_class for v in counts.values()):
            break
    return items


def extract(items):
    sift = cv2.SIFT_create()
    descs = []
    labs = []
    for gray, lab in items:
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            descs.append(np.zeros((0, 128), dtype=np.float32))
            labs.append(lab)
            continue
        if len(des) > N_DESC_CAP:
            responses = np.array([k.response for k in kp])
            top = np.argpartition(-responses, N_DESC_CAP)[:N_DESC_CAP]
            des = des[top]
        descs.append(des.astype(np.float32))
        labs.append(lab)
    return descs, np.array(labs)


def encode(per_image_desc, km, K):
    H = np.zeros((len(per_image_desc), K), dtype=np.float32)
    for i, des in enumerate(per_image_desc):
        if len(des) == 0:
            continue
        H[i] = np.bincount(km.predict(des), minlength=K)
    s = H.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return H / s


def main():
    train_ds = datasets.STL10(root=str(DATA_DIR), split="train", download=True)
    items = select_subset(train_ds, PER_CLASS_TRAIN)
    descs, y = extract(items)
    grays = [it[0] for it in items]

    counts = np.array([len(d) for d in descs])
    valid = counts >= 30  # require enough descriptors so BoVW is meaningful

    all_des = np.vstack([d for d in descs if len(d) > 0])
    km = MiniBatchKMeans(n_clusters=K_VOCAB, batch_size=1024,
                         n_init=3, random_state=0).fit(all_des)
    X = encode(descs, km, K_VOCAB)
    D = squareform(pdist(X))

    # Search every (i, j, k): i and j same class, k different. Maximize D[i,k]/D[i,j].
    # Restrict to valid (enough descriptors) images.
    n = len(X)
    best_ratio = -np.inf
    best = None
    for i in range(n):
        if not valid[i]:
            continue
        same = (y == y[i]) & valid & (np.arange(n) != i)
        diff = (y != y[i]) & valid
        if not same.any() or not diff.any():
            continue
        same_idx = np.where(same)[0]
        diff_idx = np.where(diff)[0]
        j = same_idx[np.argmin(D[i, same_idx])]
        k = diff_idx[np.argmax(D[i, diff_idx])]
        if D[i, j] < 1e-6:
            continue
        ratio = D[i, k] / D[i, j]
        if ratio > best_ratio:
            best_ratio = ratio
            best = (i, j, k)

    i, j, k = best
    print(f"\nBest triplet (by ratio):")
    print(f"  i = {i}  ({CLASS_NAMES[y[i]]})")
    print(f"  j = {j}  ({CLASS_NAMES[y[j]]})  same-class as i")
    print(f"  k = {k}  ({CLASS_NAMES[y[k]]})  different-class")
    print(f"  D(i,j) same-class    = {D[i,j]:.4f}")
    print(f"  D(i,k) different     = {D[i,k]:.4f}")
    print(f"  ratio                = {best_ratio:.2f}x")
    print(f"  descriptor counts    = i:{counts[i]}  j:{counts[j]}  k:{counts[k]}")

    # Save preview
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    titles = [
        f"i = {i}  ({CLASS_NAMES[y[i]]})",
        f"j = {j}  ({CLASS_NAMES[y[j]]})  same class\nD(i,j) = {D[i,j]:.3f}",
        f"k = {k}  ({CLASS_NAMES[y[k]]})  different class\nD(i,k) = {D[i,k]:.3f}",
    ]
    for col, idx, t in zip(range(3), [i, j, k], titles):
        axes[0, col].imshow(grays[idx], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(t)
        axes[0, col].axis("off")
        axes[1, col].bar(np.arange(K_VOCAB), X[idx], width=1.0)
        axes[1, col].set_xlim([-0.5, K_VOCAB - 0.5])
        axes[1, col].set_ylim([0, X[[i, j, k]].max() * 1.1])
        axes[1, col].set_xlabel("Visual word")
    fig.suptitle(f"BoVW triplet — same-class distance vs different-class distance "
                 f"(ratio = {best_ratio:.2f}x)")
    plt.tight_layout()
    out = ASSETS / "distance_triplet.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
