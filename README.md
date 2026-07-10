# Capstone-3

# Part 3 – Advanced Modeling

## Scenario

The objective of this part is to compare different machine learning models, evaluate their performance using cross-validation, perform hyperparameter tuning, and build a robust machine learning pipeline that can be saved and reused for future predictions.

---

# 1. Decision Tree Baseline

A Decision Tree Classifier was trained using the default parameters (`max_depth=None`).

### Results

- **Training Accuracy:** 1.0000
- **Testing Accuracy:** 0.9067

### Interpretation

The baseline Decision Tree achieved perfect training accuracy but lower testing accuracy, indicating that it learned the training data extremely well. Decision Trees are considered high-variance models because they make greedy decisions at every split and can easily memorize the training data, which may reduce their ability to generalize to unseen data.

---

# 2. Controlled Decision Tree

A second Decision Tree was trained using:

- **max_depth = 5**
- **min_samples_split = 20**

### Results

| Model | Training Accuracy | Testing Accuracy |
|-------|------------------:|-----------------:|
| Baseline Decision Tree | 1.0000 | 0.9067 |
| Controlled Decision Tree | 0.9289 | 0.9366 |

### Interpretation

The controlled Decision Tree reduced overfitting by limiting the tree depth and preventing unnecessary splits.

- **max_depth** limits how deep the tree can grow, reducing variance.
- **min_samples_split** prevents splitting nodes with very few samples, reducing noise.

The controlled model achieved better testing accuracy and improved generalization.

---

# 3. Gini vs Entropy

Two Decision Tree models were compared using different splitting criteria.

## Gini Formula

**Gini = 1 − Σ(pi²)**

## Entropy Formula

**Entropy = − Σ(pi × log₂(pi))**

If **Gini = 0**, the node is pure because all samples belong to the same class.

### Results

| Criterion | Test Accuracy |
|-----------|--------------:|
| Gini | 0.9403 |
| Entropy | 0.9254 |

### Interpretation

The Gini criterion produced slightly better test accuracy than Entropy for this dataset.

---

# 4. Random Forest

A Random Forest Classifier was trained with:

- **n_estimators = 100**
- **max_depth = 10**
- **random_state = 42**

### Results

- **Training Accuracy:** 0.9635
- **Testing Accuracy:** 0.9478
- **ROC-AUC:** 0.9487

## Top 5 Important Features

| Feature | Importance |
|---------|-----------:|
| age | 0.516106 |
| smoker_yes | 0.293830 |
| bmi | 0.110592 |
| children | 0.041152 |
| sex_male | 0.013884 |

### Interpretation

Random Forest computes feature importance using the average reduction in Gini impurity contributed by each feature across all trees.

Unlike Linear Regression coefficients, feature importance indicates how useful a feature is for splitting the data rather than showing the direction of its relationship with the target.

## Bagging

Random Forest uses **bootstrap sampling**, where each tree is trained on a random sample of the training data with replacement.

At each split, only a random subset of features is considered.

Combining predictions from many trees reduces variance and improves model stability.

---

# 4a. Gradient Boosting

Gradient Boosting was trained using:

- **n_estimators = 100**
- **learning_rate = 0.1**
- **max_depth = 3**

### Results

- **Training Accuracy:** 0.9560
- **Testing Accuracy:** 0.9328
- **ROC-AUC:** 0.9502

---

# 4b. Feature Ablation Study

The five least important features identified by the Random Forest were removed.

### Removed Features

- children
- sex_male
- region_southeast
- region_northwest
- region_southwest

### Results

| Model | ROC-AUC |
|-------|---------:|
| Full Random Forest | 0.9487 |
| Reduced Random Forest | 0.9396 |

### Interpretation

Removing the five least important features caused the ROC-AUC to decrease.

This indicates that these features still contribute useful predictive information.

Although removing features simplifies the model and reduces computation, it should only be done if performance degradation is acceptable.

---

# 5. Cross Validation

A **5-fold Stratified Cross Validation** was performed.

| Model | Mean ROC-AUC | Std ROC-AUC |
|-------|-------------:|------------:|
| Logistic Regression | 0.9467 | 0.0090 |
| Controlled Decision Tree | 0.9352 | 0.0156 |
| Random Forest | 0.9547 | 0.0078 |
| Gradient Boosting | **0.9552** | **0.0077** |

### Best Model

**Gradient Boosting**

- Mean ROC-AUC = **0.9552**
- Std ROC-AUC = **0.0077**

### Interpretation

Cross-validation provides a more reliable estimate of model performance because every sample is used for both training and validation across different folds, reducing dependence on a single train-test split.

---

# 6. GridSearchCV

## Parameter Grid

- n_estimators = [50, 100, 200]
- max_depth = [5, 10, None]
- min_samples_leaf = [1, 5]

### Best Parameters

```python
{
'randomforestclassifier__max_depth': 10,
'randomforestclassifier__min_samples_leaf': 1,
'randomforestclassifier__n_estimators': 200
}
```

### Best ROC-AUC Score

**0.9553**

### Search Space

- Parameter combinations = **18**
- Total models evaluated = **90**

### Interpretation

Grid Search evaluates every possible parameter combination and selects the one with the highest cross-validation score. Although it is computationally expensive, it guarantees evaluation of all parameter combinations, unlike Randomized Search.

---

# 7. Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|------------------:|-------------:|---------:|
| 20% | 1.0000 | 0.9591 |
| 40% | 1.0000 | 0.9492 |
| 60% | 0.99997 | 0.9464 |
| 80% | 0.99999 | 0.9506 |
| 100% | 0.99975 | 0.9515 |

### Interpretation

The Training AUC remained close to 1.0, showing that the model fits the training data extremely well. The Test AUC stabilized around 0.95 as more data was added, indicating that the model has good generalization performance. Since the Test AUC has largely plateaued, the model appears to be **capacity-limited rather than data-limited**.

---

# 8. Model Serialization

The best machine learning pipeline was successfully saved as:

```text
best_model.pkl
```

The saved model was successfully reloaded using **joblib.load()**.

Predictions generated for two unseen samples were:

```text
[1 0]
```

This confirms that the serialized model can be reused without retraining.

---

# Final Model Comparison

| Model | Test ROC-AUC | 5-Fold Mean ROC-AUC | 5-Fold Std ROC-AUC |
|-------|-------------:|--------------------:|-------------------:|
| Logistic Regression | 0.9529 | 0.9467 | 0.0090 |
| Controlled Decision Tree | 0.9403 | 0.9352 | 0.0156 |
| Random Forest | 0.9487 | 0.9547 | 0.0078 |
| Gradient Boosting | **0.9502** | **0.9552** | **0.0077** |

---

# Recommended Model

Based on the experimental results, **Gradient Boosting** is recommended because it achieved the highest average 5-fold cross-validation ROC-AUC (**0.9552**) while maintaining the lowest standard deviation (**0.0077**), demonstrating consistent performance across different validation folds. It also achieved a high test ROC-AUC (**0.9502**), indicating strong predictive performance on unseen data. Overall, Gradient Boosting provides the best balance between accuracy, robustness, and generalization, making it the most suitable model for deployment.
