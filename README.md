
# Breast Cancer Classification Using Decision Tree and Random Forest: A Comparative Study.

This project uses the Wisconsin Breast Cancer Dataset to apply and compare the performance of Decision Tree and Random Forest classifiers. The goal is to predict whether a tumor is benign or malignant based on various continuous attributes.
## Dataset Description

The Wisconsin Breast Cancer Dataset consists of 10 continuous attributes and 1 target class attribute:

* Attributes: 10 continuous features related to tumor characteristics.
* Target Class: Diagnosis ('B' for benign and 'M' for malignant).

## Steps Followed in the Project
### Data Loading and Preprocessing:

* Loaded the dataset from a CSV file.
* Removed the irrelevant Id column.
* Scaled the features using Z-scores for normalization.
### Model Building:

* Split the data into training and testing sets (70/30 split).
* Applied Decision Tree and Random Forest classifiers.
### Model Evaluation:

* Calculated and displayed performance metrics including accuracy, confusion matrix, and classification report for both models.
* Used cross-validation to compare the accuracy of both algorithms.
### Results:

* Accuracy, confusion matrix, and classification report for each model were displayed.
* A boxplot was generated to compare the accuracy scores of both models.

## Requirements
To run this project, you need the following libraries:

1. numpy
2. pandas
3. matplotlib
4. seaborn
5. scikit-learn

*You can install them using the following:*

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Code Walkthrough
### Step 1: Import Libraries

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Step 2: Load and Explore the Dataset
```bash
data = pd.read_csv('/content/wisc_bc_data.csv')
data.head()
```
### Step 3: Data Preprocessing
* Drop the **Id** column.
* Scale features using Z-scores.

```bash
features = data.drop('diagnosis', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
```

### Step 4: Split the Data into Training and Testing Sets

```bash
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
```

### Step 5: Train and Evaluate Models

* Train and evaluate the Decision Tree and Random Forest models.
* Generate accuracy, confusion matrix, and classification report.

```bash
for name, model in algorithms:
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    results[name] = {'accuracy': accuracy, 'confusion_matrix': cm, 'classification_report': report}
```
### Step 6: Cross-validation and Boxplot Comparison

```bash
for name, model in algorithms:
    scores = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
    results[name] = scores

flat_accuracy_scores = [score for name in results for score in results[name]]
flat_algorithms = [name for name in results for _ in results[name]]

data_for_plot = pd.DataFrame({'Algorithm': flat_algorithms, 'Accuracy': flat_accuracy_scores})
sns.boxplot(x='Algorithm', y='Accuracy', data=data_for_plot)
```
## Results
The results for Decision Tree and Random Forest classifiers include:

* Accuracy
* Confusion Matrix
* Classification Report
Additionally, a **boxplot** is generated to visually compare the accuracy distribution of both models.

## Conclusion
The project demonstrates how Decision Tree and Random Forest classifiers perform on the Wisconsin Breast Cancer Dataset and compares their accuracy through cross-validation. Based on the metrics, you can select the best performing model for further analysis or deployment.
