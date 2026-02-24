# Advanced Machine Learning (ETH, 2025)

This repository contains our final project for the Advanced Machine Learning course at ETH Zurich (Fall 2025).

The project was part of a Kaggle competition involving all course students. Our team finished 3rd out of 136 teams, with a public score around R² = 0.78.

## Project Summary

We built a full regression pipeline with outlier removal, feature selection, robust preprocessing, and stacked ensembling.

### 1) Outlier Detection

First, we removed the outliers from the training set. We first did a temporary median imputation, PCA projection to 2 dimensions, and finally Isolation Forest on the 2D projections. This removed around 4.7% of the training samples.

### 2) Feature Selection

We ranked features by absolute Spearman correlation with the target, kept the top informative subset (around 200 features) and refined with RandomForest feature importances, keeping the strongest final subset (around 180 features).

### 3) Preprocessing

For each model branch:
- standardization using training-set statistics
- missing value handling with KNN imputation (k=2) (Iterative Imputer was also tested and provided similar results)
KNN imputation preserved local structure better than simple median imputation in our experiments.

### 4) Ensemble Modeling

We used a stacking strategy with complementary regressors.

The models we used in the final submission, that are present in this codebase, were:

- Support Vector Regressor (SVR)
- Gaussian Process Regressor (RationalQuadratic + White kernels)
- HistGradientBoosting Regressor

During the competition, we also tested other learners (including XGBoost, ExtraTrees Regressor, LightGBM, KNN, and MLP). They were slightly less effective in our final setup.

The meta-learner combines base model predictions with a linear model (LinearRegression in this repository; ElasticNet and Ridge were also tested during the project).

### 5) Validation Strategy

We evaluated performance with 10-fold cross-validation, reporting mean R² and standard deviation on validation folds.


## Repository Structure

To run cross validation:

```bash
python stack.py
```

To produce the final predictions:

```bash
python main.py
```