# Rice Variety Classification with Random Forests

A machine learning project that classifies rice grains as **Cammeo** or **Osmancik** based on their geometric properties, using a tuned Random Forest classifier.

## Overview

Manually sorting rice by variety is impractical at scale, but it matters for quality control and market value. This project investigates whether simple geometric measurements taken from grain images — area, perimeter, axis lengths, eccentricity — are enough to reliably tell the two varieties apart.

This is a binary classification problem, tackled with an ensemble tree-based approach (Random Forest), including exploratory data analysis, model evaluation, and hyperparameter tuning to address overfitting.

## Dataset

**Source:** [Rice (Cammeo and Osmancik) — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)

- 3,810 samples, 7 numerical features, 1 target class
- No missing values
- Classes are reasonably balanced (Osmancik slightly more frequent)
- Included in this repo as `rice_data.csv`

| Feature | Description |
|---|---|
| `Area` | Size of the rice grain |
| `Perimeter` | Distance around the outer edge of the grain |
| `Major_Axis_Length` | Length along the grain's longest direction |
| `Minor_Axis_Length` | Width along the grain's shortest direction |
| `Eccentricity` | How elongated the grain is |
| `Convex_Area` | Area of the smallest convex shape enclosing the grain |
| `Extent` | Ratio of grain area to bounding box area |

## Approach

1. **Exploratory analysis** — summary statistics, boxplots, correlation heatmap, and per-class density plots to understand feature distributions and separability.
2. **Feature selection** — dropped `Extent` (weak class separation) and `Convex_Area` (near-perfect correlation with `Area`), keeping five features.
3. **Modelling** — trained a Random Forest classifier on an 80/20 stratified train-test split.
4. **Evaluation** — accuracy, precision, recall, confusion matrix, ROC/AUC, and 5-fold stratified cross-validation to get a more reliable read on generalisation.
5. **Tuning** — used learning curves to diagnose overfitting, then `GridSearchCV` to tune `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

## Results

| Metric | Default Model | Tuned Model |
|---|---|---|
| Test Accuracy | 0.913 | improved |
| Test Precision | 0.919 | slightly lower |
| Test Recall | 0.931 | 0.945 |
| Cross-val Test Accuracy | 0.920 ± 0.007 | 0.929 |
| ROC AUC | 0.97 | — |

The default Random Forest memorised the training set almost perfectly (train accuracy of 1.0), which pointed to overfitting from the unrestricted tree depth. Constraining tree depth and split/leaf sizes via grid search closed that gap: cross-validated accuracy rose from 0.920 to 0.929, and recall improved from 0.931 to 0.945, at a small cost to precision. Size-related features (`Area`, `Perimeter`, `Major_Axis_Length`) turned out to be the strongest signals for telling the two varieties apart.

## Project Structure
