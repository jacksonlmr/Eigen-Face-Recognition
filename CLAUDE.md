# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `opencv-python`, `matplotlib`

## Running the Pipeline

**Step 1 — Training** (computes eigenfaces from `Faces_FA_FB/fa_H/`):
```bash
python training.py
```
Outputs saved to disk: `avg_face.npy`, `eigen_values.npy`, `eigen_vectors.npy`, `eigen_coef.npy`, `training_labels.npy`, and visualizations in `eigen_faces_imgs/`.

**Step 2 — Testing** (face recognition on a test directory):
```bash
python testing.py <test_data_path> [-t <info_threshold>]
# Example:
python testing.py Faces_FA_FB/fb_H -t 0.9
```
`-t` controls what proportion of variance to retain when selecting how many eigenfaces to use (default 0.9). Outputs a Top-r accuracy plot saved as `Experiment_II.jpg`.

**Reconstruction verification** (sanity-check eigenface decomposition):
```bash
python recon_verify.py
```

## Architecture

The system implements PCA-based face recognition (eigenfaces) in two phases:

**Training (`training.py`)**
1. Loads grayscale images from a directory via `helpers.load_images()`, which returns flattened pixel arrays (each image is one row), label filenames, and the original image shape.
2. Computes the mean face and centers the data matrix.
3. Uses the covariance trick: computes `A^T * A` (n×n, where n = number of images) instead of `A * A^T` (p×p, where p = pixels), then projects eigenvectors back to pixel space via `A * v`.
4. Normalizes eigenvectors and saves all artifacts to `.npy` files.
5. Projects each training image into eigenface space to store `eigen_coef.npy` (one coefficient vector per training image).

**Testing (`testing.py`)**
1. Selects the number of eigenfaces to keep by finding where cumulative explained variance exceeds the threshold.
2. Projects test images into the truncated eigenface space.
3. Identifies the nearest training image using **Mahalanobis distance** (weighted by eigenvalues, not plain Euclidean).
4. Reports Top-r recognition accuracy and plots the curve via `plot.plot_accuracies()`.

**Key data shapes** (with N training images, p pixels, k kept eigenfaces):
- `eigen_vectors.npy`: `(p, N)` — columns are eigenfaces in pixel space
- `eigen_coef.npy`: `(N, N)` — row i is the eigenspace projection of training image i
- At test time, both matrices are sliced to `[:, :k]` based on the variance threshold

## Dataset

Images live in `Faces_FA_FB/` (`.pgm` grayscale format). Training uses `fa_H/`; testing typically uses `fb_H/`. The dataset is also provided as `Faces_FA_FB.zip`.
