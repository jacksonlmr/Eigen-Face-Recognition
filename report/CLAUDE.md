# Report Context: Programming Assignment 3

## LaTeX Configuration
The `report.tex` file uses the following document class and packages:
- **Document Class:** `article`
- **Core Packages:** `geometry`, `inputenc`, `fontenc`, `babel`, `graphicx`
- **Typography:** `helvet` (Sans Serif / Helvetica)
- **Mathematical & Tabular:** `amsmath`, `amssymb`, `booktabs`, `xcolor`, `colortbl`, `float`, `subcaption`
- **Custom Definitions:** Includes several custom colors for table cell highlighting (`trueyellow`, `meangreen`, `sigmablue`, `offpurple`, etc.) and legend colors for misclassification (`negchange`, `improved`, `greatimproved`, `worsened`, `greatworsened`).

## Report Information
- **Title:** Programming Assignment 3: Eigenfaces for Recognition
- **Course:** CS 479/679 Pattern Recognition
- **Author:** Jackson Loughmiller
- **Date:** May 4, 2026
- **Summary:** The report covers the theory and implementation of PCA-based face recognition (eigenfaces) using the FERET face database. It includes a Theory section explaining dimensionality reduction, feature extraction, and PCA, followed by Results and Discussion covering the following experiments:
  - **Experiment (a):** Training on `fa_H` (1204 images), testing on `fb_H` (1196 images). Shows the average face and top/bottom 10 eigenfaces. Computes CMC curves using Mahalanobis distance at 80%, 90%, and 95% variance thresholds. Shows 3 correctly and 3 incorrectly matched query images at r=1.
  - **Experiment (b):** Intruder detection — removes first 50 subjects from `fa_H` to form `fa2_H`, trains on `fa2_H`, tests on `fb_H`. Plots an ROC curve (True Positive Rate vs. False Positive Rate) by varying the distance threshold T_r.
  - **Experiments (c)–(e) (Extra Credit):** Repeat (a) and (b) using low-resolution images (`fa_L`, `fb_L`, `fa2_L`) and analyze the effect of image resolution on recognition performance.

## Report Structure
- **Title page** with declaration statement
- **Theory**
  - Experiment a: Face Recognition
    - Dimensionality Reduction
    - Feature Extraction
    - Principal Component Analysis
- **Results and Discussion**
  - Experiment 1 (corresponds to assignment experiment a)
  - Experiment 2 (corresponds to assignment experiment b)

## Critical Instructions for Claude
- **DO NOT** edit any text that has already been written, unless instructed to do so.
- **ONLY** perform the requests the user gave explicitly. If you think more is needed, you **MUST** ask the user first.
- The user is required to write their own work for this report. You are only permitted to change things such as tables, equations, and other visuals and formatting for the user.
