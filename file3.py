Applied AI & ML Essentials Capstone
Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
Student Name: Aashika R
Dataset: Medical Cost Personal Dataset
Download the medical cost personal dataset from Kaggle
------------------------------------------------------------------------
# TASK 1 - DECISION TREE BASELINE

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("="*60)
print("TASK 1 - DECISION TREE BASELINE")
print("="*60)

# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train model
dt_model.fit(X_train_clf_scaled, y_clf_train)

# Predictions
train_pred = dt_model.predict(X_train_clf_scaled)
test_pred = dt_model.predict(X_test_clf_scaled)

# Accuracy
train_accuracy = accuracy_score(y_clf_train, train_pred)
test_accuracy = accuracy_score(y_clf_test, test_pred)

print("\nTraining Accuracy :", round(train_accuracy,4))
print("Testing Accuracy  :", round(test_accuracy,4))

# Overfitting Check
print("\n")
print("="*60)
print("OVERFITTING CHECK")
print("="*60)

if train_accuracy - test_accuracy > 0.10:
    print("The Decision Tree appears to be overfitting.")
    print("Training accuracy is much higher than testing accuracy.")
else:
    print("No significant overfitting detected.")
  Output:
============================================================
TASK 1 - DECISION TREE BASELINE
============================================================

Training Accuracy : 1.0
Testing Accuracy  : 0.9067


============================================================
OVERFITTING CHECK
============================================================
No significant overfitting detected.

------------------------------------------------------------------------------
# TASK 2 - CONTROLLED DECISION TREE

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("="*60)
print("TASK 2 - CONTROLLED DECISION TREE")
print("="*60)

# Create Controlled Decision Tree
controlled_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

# Train Model
controlled_tree.fit(X_train_clf_scaled, y_clf_train)

# Predictions
train_pred_controlled = controlled_tree.predict(X_train_clf_scaled)
test_pred_controlled = controlled_tree.predict(X_test_clf_scaled)

# Accuracy
train_accuracy_controlled = accuracy_score(
    y_clf_train,
    train_pred_controlled
)

test_accuracy_controlled = accuracy_score(
    y_clf_test,
    test_pred_controlled
)

print("\nTraining Accuracy :", round(train_accuracy_controlled,4))
print("Testing Accuracy  :", round(test_accuracy_controlled,4))

# Compare with Baseline Decision Tree

print("\n")
print("="*60)
print("COMPARISON")
print("="*60)

comparison = pd.DataFrame({

    "Model":[
        "Baseline Decision Tree",
        "Controlled Decision Tree"
    ],

    "Training Accuracy":[
        train_accuracy,
        train_accuracy_controlled
    ],

    "Testing Accuracy":[
        test_accuracy,
        test_accuracy_controlled
    ]

})

print(comparison)

# Train-Test Gap

baseline_gap = train_accuracy - test_accuracy
controlled_gap = train_accuracy_controlled - test_accuracy_controlled

print("\nBaseline Gap  :", round(baseline_gap,4))
print("Controlled Gap:", round(controlled_gap,4))

if controlled_gap < baseline_gap:
    print("\nControlled Decision Tree reduces overfitting.")
else:
    print("\nControlled Decision Tree does not reduce overfitting.")

Output:
============================================================
TASK 2 - CONTROLLED DECISION TREE
============================================================

Training Accuracy : 0.9289
Testing Accuracy  : 0.9366


============================================================
COMPARISON
============================================================
                      Model  Training Accuracy  Testing Accuracy
0    Baseline Decision Tree           1.000000          0.906716
1  Controlled Decision Tree           0.928906          0.936567

Baseline Gap  : 0.0933
Controlled Gap: -0.0077

Controlled Decision Tree reduces overfitting.
--------------------------------------------------------------------------
# TASK 3 - GINI VS ENTROPY COMPARISON

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("="*60)
print("TASK 3 - GINI VS ENTROPY COMPARISON")
print("="*60)

# Decision Tree using Gini

gini_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

gini_tree.fit(X_train_clf_scaled, y_clf_train)

gini_pred = gini_tree.predict(X_test_clf_scaled)

gini_accuracy = accuracy_score(y_clf_test, gini_pred)

# Decision Tree using Entropy

entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

entropy_tree.fit(X_train_clf_scaled, y_clf_train)

entropy_pred = entropy_tree.predict(X_test_clf_scaled)

entropy_accuracy = accuracy_score(y_clf_test, entropy_pred)

# Print Results

print("\nGini Test Accuracy    :", round(gini_accuracy,4))

print("Entropy Test Accuracy :", round(entropy_accuracy,4))

# Comparison Table

comparison = pd.DataFrame({

    "Criterion":[
        "Gini",
        "Entropy"
    ],

    "Test Accuracy":[
        gini_accuracy,
        entropy_accuracy
    ]

})
print("="*60)
print("GINI VS ENTROPY")
print("="*60)

print(comparison)


print("\n")
# Better Criterion

if gini_accuracy > entropy_accuracy:
    print("\nGini criterion performed better.")
elif entropy_accuracy > gini_accuracy:
    print("\nEntropy criterion performed better.")
else:
    print("\nBoth criteria achieved the same accuracy.")

Output:
============================================================
TASK 3 - GINI VS ENTROPY COMPARISON
============================================================

Gini Test Accuracy    : 0.9403
Entropy Test Accuracy : 0.9254


============================================================
GINI VS ENTROPY
============================================================
  Criterion  Test Accuracy
0      Gini       0.940299
1   Entropy       0.925373

Gini criterion performed better
---------------------------------------------------------------------
# TASK 4 - RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

print("="*60)
print("TASK 4 - RANDOM FOREST CLASSIFIER")
print("="*60)

# Train Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train_clf_scaled, y_clf_train)

# Predictions

train_pred_rf = rf_model.predict(X_train_clf_scaled)

test_pred_rf = rf_model.predict(X_test_clf_scaled)

test_prob_rf = rf_model.predict_proba(X_test_clf_scaled)[:,1]

# Accuracy

train_accuracy_rf = accuracy_score(
    y_clf_train,
    train_pred_rf
)

test_accuracy_rf = accuracy_score(
    y_clf_test,
    test_pred_rf
)

auc_rf = roc_auc_score(
    y_clf_test,
    test_prob_rf
)

print("\nTraining Accuracy :", round(train_accuracy_rf,4))

print("Testing Accuracy  :", round(test_accuracy_rf,4))

print("ROC-AUC Score     :", round(auc_rf,4))

# Feature Importance

feature_importance = pd.DataFrame({

    "Feature": X_train_clf.columns,

    "Importance": rf_model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")

print("="*60)
print("TOP 5 IMPORTANT FEATURES")
print("="*60)

print(feature_importance.head(5))

# Save Top 5 Features (Used in Task 4b)

top5_features = feature_importance.head(5)

# Save Lowest 5 Features (Used in Task 4b)

lowest5_features = feature_importance.tail(5)

print("\nLowest 5 Features")

print(lowest5_features)

Output:
============================================================
TASK 4 - RANDOM FOREST CLASSIFIER
============================================================

Training Accuracy : 0.9635
Testing Accuracy  : 0.9478
ROC-AUC Score     : 0.9487


============================================================
TOP 5 IMPORTANT FEATURES
============================================================
      Feature  Importance
0         age    0.516106
4  smoker_yes    0.293830
1         bmi    0.110592
2    children    0.041152
3    sex_male    0.013884

Lowest 5 Features
            Feature  Importance
2          children    0.041152
3          sex_male    0.013884
6  region_southeast    0.010325
5  region_northwest    0.007155
7  region_southwest    0.006957
----------------------------------------------------------------
# TASK 4a - GRADIENT BOOSTING CLASSIFIER

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

print("="*60)
print("TASK 4a - GRADIENT BOOSTING CLASSIFIER")
print("="*60)

# Train Gradient Boosting Model

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train_clf_scaled, y_clf_train)

# Predictions

train_pred_gb = gb_model.predict(X_train_clf_scaled)

test_pred_gb = gb_model.predict(X_test_clf_scaled)

test_prob_gb = gb_model.predict_proba(X_test_clf_scaled)[:,1]

# Evaluation
train_accuracy_gb = accuracy_score(
    y_clf_train,
    train_pred_gb
)

test_accuracy_gb = accuracy_score(
    y_clf_test,
    test_pred_gb
)

auc_gb = roc_auc_score(
    y_clf_test,
    test_prob_gb
)

print("\nTraining Accuracy :", round(train_accuracy_gb,4))
print("Testing Accuracy  :", round(test_accuracy_gb,4))
print("ROC-AUC Score     :", round(auc_gb,4))

Output:
============================================================
TASK 4a - GRADIENT BOOSTING CLASSIFIER
============================================================

Training Accuracy : 0.956
Testing Accuracy  : 0.9328
ROC-AUC Score     : 0.9502
-------------------------------------------------------------------
# TASK 4b - FEATURE ABLATION STUDY

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

print("="*60)
print("TASK 4b - FEATURE ABLATION STUDY")
print("="*60)

# Identify Lowest 5 Features

lowest_features = feature_importance.tail(5)["Feature"].tolist()

print("\nLowest 5 Features:")

for feature in lowest_features:
    print(feature)

# Remove Lowest Features

X_train_reduced = X_train_clf.drop(columns=lowest_features)

X_test_reduced = X_test_clf.drop(columns=lowest_features)

# Scale Reduced Features

from sklearn.preprocessing import StandardScaler

scaler_reduced = StandardScaler()

X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)

X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)

# Train Random Forest on Reduced Dataset

rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_reduced.fit(
    X_train_reduced_scaled,
    y_clf_train
)

# Prediction

reduced_prob = rf_reduced.predict_proba(
    X_test_reduced_scaled
)[:,1]

auc_reduced = roc_auc_score(
    y_clf_test,
    reduced_prob
)

# Compare AUC

comparison = pd.DataFrame({

    "Model":[
        "Full Random Forest",
        "Reduced Random Forest"
    ],

    "ROC-AUC":[
        auc_rf,
        auc_reduced
    ]

})

print("\n")

print("="*60)
print("AUC COMPARISON")
print("="*60)

print(comparison)

# Interpretation

if auc_reduced >= auc_rf:
    print("\nRemoving the least important features did not reduce performance.")
    print("These features contribute little information.")

else:
    print("\nPerformance decreased after removing the features.")
    print("The removed features still contribute useful information.")
  
Output:
============================================================
TASK 4b - FEATURE ABLATION STUDY
============================================================

Lowest 5 Features:
children
sex_male
region_southeast
region_northwest
region_southwest


============================================================
AUC COMPARISON
============================================================
                   Model   ROC-AUC
0     Full Random Forest  0.948708
1  Reduced Random Forest  0.939630

Performance decreased after removing the features.
The removed features still contribute useful information.
--------------------------------------------------------------------
# TASK 5 - CROSS VALIDATION COMPARISON
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd

print("="*60)
print("TASK 5 - CROSS VALIDATED MODEL COMPARISON")
print("="*60)

# Cross Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Logistic Regression

logistic_scores = cross_val_score(
    logistic_model,
    X_train_clf_scaled,
    y_clf_train,
    cv=cv,
    scoring="roc_auc"
)

# Controlled Decision Tree

tree_scores = cross_val_score(
    controlled_tree,
    X_train_clf_scaled,
    y_clf_train,
    cv=cv,
    scoring="roc_auc"
)

# Random Forest

rf_scores = cross_val_score(
    rf_model,
    X_train_clf_scaled,
    y_clf_train,
    cv=cv,
    scoring="roc_auc"
)

# Gradient Boosting

gb_scores = cross_val_score(
    gb_model,
    X_train_clf_scaled,
    y_clf_train,
    cv=cv,
    scoring="roc_auc"
)

# Comparison Table

comparison = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Controlled Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Mean ROC-AUC":[
        logistic_scores.mean(),
        tree_scores.mean(),
        rf_scores.mean(),
        gb_scores.mean()
    ],

    "Std ROC-AUC":[
        logistic_scores.std(),
        tree_scores.std(),
        rf_scores.std(),
        gb_scores.std()
    ]

})

comparison = comparison.sort_values(
    by="Mean ROC-AUC",
    ascending=False
)

print("\n")

print("="*60)
print("5-FOLD CROSS VALIDATION RESULTS")
print("="*60)

print(comparison)

# Best Model

best_model = comparison.iloc[0]

print("\n")

print("="*60)
print("BEST MODEL")
print("="*60)

print("Model :", best_model["Model"])

print("Mean ROC-AUC :", round(best_model["Mean ROC-AUC"],4))

print("Std ROC-AUC :", round(best_model["Std ROC-AUC"],4))

Output:
============================================================
TASK 5 - CROSS VALIDATED MODEL COMPARISON
============================================================


============================================================
5-FOLD CROSS VALIDATION RESULTS
============================================================
                      Model  Mean ROC-AUC  Std ROC-AUC
3         Gradient Boosting      0.955168     0.007657
2             Random Forest      0.954664     0.007751
0       Logistic Regression      0.946726     0.008959
1  Controlled Decision Tree      0.935218     0.015576


============================================================
BEST MODEL
============================================================
Model : Gradient Boosting
Mean ROC-AUC : 0.9552
Std ROC-AUC : 0.0077
-------------------------------------------------------------------
# TASK 6 - GRID SEARCH WITH PIPELINE

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

print("="*60)
print("TASK 6 - GRID SEARCH CV")
print("="*60)

# Build Pipeline

pipeline = make_pipeline(

    SimpleImputer(strategy="median"),

    StandardScaler(),

    RandomForestClassifier(random_state=42)

)

# Parameter Grid

param_grid = {

    "randomforestclassifier__n_estimators":[50,100,200],

    "randomforestclassifier__max_depth":[5,10,None],

    "randomforestclassifier__min_samples_leaf":[1,5]

}

# Cross Validation
cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

# Grid Search

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=cv,

    n_jobs=-1

)

# Train Grid Search

grid_search.fit(X_train_clf, y_clf_train)

# Best Parameters

print("\n")

print("="*60)
print("BEST PARAMETERS")
print("="*60)

print(grid_search.best_params_)

print("\nBest ROC-AUC :", round(grid_search.best_score_,4))

# Number of Models Evaluated

total_models = (

    len(param_grid["randomforestclassifier__n_estimators"])

    *

    len(param_grid["randomforestclassifier__max_depth"])

    *

    len(param_grid["randomforestclassifier__min_samples_leaf"])

)

print("\nTotal Parameter Combinations :", total_models)

print("Total Models Trained :", total_models * 5)

# Best Pipeline
best_pipeline = grid_search.best_estimator_

Output:
============================================================
TASK 6 - GRID SEARCH CV
============================================================


============================================================
BEST PARAMETERS
============================================================
{'randomforestclassifier__max_depth': 10, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 200}

Best ROC-AUC : 0.9553

Total Parameter Combinations : 18
Total Models Trained : 90

----------------------------------------------------------------------------
# TASK 7 - MANUAL LEARNING CURVE

from sklearn.metrics import roc_auc_score
import pandas as pd

print("="*60)
print("TASK 7 - MANUAL LEARNING CURVE")
print("="*60)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

results = []

for f in fractions:

    n = int(f * len(X_train_clf))

    X_subset = X_train_clf.iloc[:n]

    y_subset = y_clf_train.iloc[:n]

    # Train best pipeline
    best_pipeline.fit(X_subset, y_subset)

    # Training AUC
    train_prob = best_pipeline.predict_proba(X_subset)[:,1]

    train_auc = roc_auc_score(
        y_subset,
        train_prob
    )

    # Test AUC
    test_prob = best_pipeline.predict_proba(X_test_clf)[:,1]

    test_auc = roc_auc_score(
        y_clf_test,
        test_prob
    )

    results.append([
        f,
        train_auc,
        test_auc
    ])

# Learning Curve Table

learning_curve = pd.DataFrame(

    results,

    columns=[
        "Training Fraction",
        "Training AUC",
        "Test AUC"
    ]

)

print("\n")

print("="*60)
print("LEARNING CURVE")
print("="*60)

print(learning_curve)

Output:

============================================================
TASK 7 - MANUAL LEARNING CURVE
============================================================


============================================================
LEARNING CURVE
============================================================
   Training Fraction  Training AUC  Test AUC
0                0.2      1.000000  0.959067
1                0.4      1.000000  0.949237
2                0.6      0.999971  0.946425
3                0.8      0.999995  0.950601
4                1.0      0.999748  0.951493
-----------------------------------------------------------------------
# TASK 8 - SAVE AND RELOAD BEST MODEL

import joblib

print("="*60)
print("TASK 8 - SAVE AND RELOAD MODEL")
print("="*60)

# Save Best Pipeline

joblib.dump(best_pipeline, "best_model.pkl")

print("\nModel saved successfully as 'best_model.pkl'.")

# Load Saved Model

loaded_model = joblib.load("best_model.pkl")

print("Model loaded successfully.")

# Predict on Two Test Samples

sample_data = X_test_clf.iloc[:2]

predictions = loaded_model.predict(sample_data)

print("\n")

print("="*60)
print("PREDICTIONS")
print("="*60)

print("Input Samples:")
print(sample_data)

print("\nPredicted Classes:")
print(predictions)

Output:

============================================================
TASK 8 - SAVE AND RELOAD MODEL
============================================================

Model saved successfully as 'best_model.pkl'.
Model loaded successfully.


============================================================
PREDICTIONS
============================================================
Input Samples:
      age    bmi  children  sex_male  smoker_yes  region_northwest  \
1182   48  27.36         1     False       False             False   
1137   33  30.25         0      True       False             False   

      region_southeast  region_southwest  
1182             False             False  
1137              True             False  

Predicted Classes:
[1 0]
-----------------------------------------------------------------------------
