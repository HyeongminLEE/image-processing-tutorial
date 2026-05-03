"""Pre-experiment for week 9 lab decision-boundary visualization.

Builds the STL-10 / SIFT / BoVW / SVM pipeline twice and saves two PNGs:
    (A) SVM trained in PCA 2D space (boundary is exact in the visualized space)
    (B) SVM trained in full BoVW; boundary visualized via PCA inverse_transform
        of a mesh grid (looks cleaner but is an approximation)

Run:
    uv run --with opencv-python --with scikit-learn --with torchvision \
        --with numpy --with matplotlib --with pillow \
        python3 scripts/explore_decision_boundary.py
"""

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.datasets as datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent  # .../src/week9
DATA_DIR = Path.home() / ".cache" / "stl10"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# 5-class subset of STL-10 (alphabetical indices).
# vehicles + animals for visual contrast.
SELECTED = {0: "airplane", 1: "bird", 2: "car", 3: "cat", 8: "ship"}
PER_CLASS_TRAIN = 100
PER_CLASS_TEST = 30
N_DESC_CAP = 100  # cap descriptors per image (top-N by response)
K_VOCAB = 200


def load_stl10(split):
    print(f"[stl10] loading {split} split (cache: {DATA_DIR})...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return datasets.STL10(root=str(DATA_DIR), split=split, download=True)


def select_subset(ds, n_per_class):
    """Return list of (gray_uint8, remapped_label)."""
    items = []
    counts = {c: 0 for c in SELECTED}
    new_label = {c: i for i, c in enumerate(SELECTED.keys())}
    for i in range(len(ds)):
        img, label = ds[i]
        if label not in SELECTED:
            continue
        if counts[label] >= n_per_class:
            continue
        gray = np.array(img.convert("L"))  # uint8 (96, 96)
        items.append((gray, new_label[label]))
        counts[label] += 1
        if all(v >= n_per_class for v in counts.values()):
            break
    return items


def extract_descriptors(items):
    sift = cv2.SIFT_create()
    per_image_desc = []
    labels = []
    for gray, lab in items:
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            per_image_desc.append(np.zeros((0, 128), dtype=np.float32))
            labels.append(lab)
            continue
        if len(des) > N_DESC_CAP:
            responses = np.array([k.response for k in kp])
            top = np.argpartition(-responses, N_DESC_CAP)[:N_DESC_CAP]
            des = des[top]
        per_image_desc.append(des.astype(np.float32))
        labels.append(lab)
    return per_image_desc, np.array(labels)


def build_vocab(per_image_desc, K):
    all_des = np.vstack([d for d in per_image_desc if len(d) > 0])
    print(f"[kmeans] all_des shape = {all_des.shape}, K = {K}")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, batch_size=1024, n_init=3, random_state=0)
    km.fit(all_des)
    print(f"[kmeans] fit in {time.time() - t0:.1f}s")
    return km


def encode_bovw(per_image_desc, km, K):
    H = np.zeros((len(per_image_desc), K), dtype=np.float32)
    for i, des in enumerate(per_image_desc):
        if len(des) == 0:
            continue
        words = km.predict(des)
        H[i] = np.bincount(words, minlength=K)
    sums = H.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return H / sums


def _scatter_with_boundary(ax, xx, yy, Z, X_tr2, y_train, class_names):
    ax.contourf(
        xx, yy, Z,
        levels=np.arange(-0.5, len(class_names) + 0.5, 1),
        cmap="tab10", alpha=0.25,
    )
    for cls in range(len(class_names)):
        mask = y_train == cls
        ax.scatter(
            X_tr2[mask, 0], X_tr2[mask, 1],
            s=20, color=plt.cm.tab10(cls),
            label=class_names[cls],
            edgecolor="k", linewidth=0.4,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)


def plot_path_a(X_train, y_train, X_test, y_test, class_names, out_path):
    pca = PCA(n_components=2, random_state=0).fit(X_train)
    X_tr2 = pca.transform(X_train)
    X_te2 = pca.transform(X_test)
    clf = SVC(kernel="rbf", gamma="scale", C=10).fit(X_tr2, y_train)
    tr_acc = clf.score(X_tr2, y_train)
    te_acc = clf.score(X_te2, y_test)

    pad = 0.3
    xx, yy = np.meshgrid(
        np.linspace(X_tr2[:, 0].min() - pad, X_tr2[:, 0].max() + pad, 250),
        np.linspace(X_tr2[:, 1].min() - pad, X_tr2[:, 1].max() + pad, 250),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_with_boundary(ax, xx, yy, Z, X_tr2, y_train, class_names)
    ax.set_title(
        f"(A) SVM trained in PCA 2D space\n"
        f"train acc = {tr_acc:.2f},  test acc = {te_acc:.2f}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"[A] saved {out_path}")


def plot_path_b(X_train, y_train, X_test, y_test, class_names, out_path):
    pca = PCA(n_components=2, random_state=0).fit(X_train)
    X_tr2 = pca.transform(X_train)
    clf = SVC(kernel="rbf", gamma="scale", C=10).fit(X_train, y_train)
    tr_acc = clf.score(X_train, y_train)
    te_acc = clf.score(X_test, y_test)

    pad = 0.3
    xx, yy = np.meshgrid(
        np.linspace(X_tr2[:, 0].min() - pad, X_tr2[:, 0].max() + pad, 250),
        np.linspace(X_tr2[:, 1].min() - pad, X_tr2[:, 1].max() + pad, 250),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_full = pca.inverse_transform(grid_2d)
    Z = clf.predict(grid_full).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_with_boundary(ax, xx, yy, Z, X_tr2, y_train, class_names)
    ax.set_title(
        f"(B) SVM trained in full BoVW (K={K_VOCAB});\n"
        f"boundary via PCA inverse_transform of 2D mesh\n"
        f"train acc = {tr_acc:.2f},  test acc = {te_acc:.2f}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"[B] saved {out_path}")


def main():
    train_ds = load_stl10("train")
    test_ds = load_stl10("test")
    train_items = select_subset(train_ds, PER_CLASS_TRAIN)
    test_items = select_subset(test_ds, PER_CLASS_TEST)
    print(f"train images = {len(train_items)},  test images = {len(test_items)}")

    print("[sift] extracting train descriptors...")
    train_desc, y_train = extract_descriptors(train_items)
    print("[sift] extracting test descriptors...")
    test_desc, y_test = extract_descriptors(test_items)

    km = build_vocab(train_desc, K_VOCAB)
    X_train = encode_bovw(train_desc, km, K_VOCAB)
    X_test = encode_bovw(test_desc, km, K_VOCAB)
    print(f"X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")

    class_names = list(SELECTED.values())
    plot_path_a(X_train, y_train, X_test, y_test, class_names,
                ASSETS / "boundary_pca2d.png")
    plot_path_b(X_train, y_train, X_test, y_test, class_names,
                ASSETS / "boundary_full_proj.png")


if __name__ == "__main__":
    main()
