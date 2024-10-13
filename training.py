import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset (Diabetes dataset)
data = load_diabetes()
X = data.data
y = data.target

# Convert target to a binary classification problem (e.g., 0 if diabetes score < 140, 1 otherwise)
y = (y >= 140).astype(int)  # Adjust the threshold based on your requirements

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers with hyperparameters
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),  # Increase max_iter for convergence
    'DecisionTree': DecisionTreeClassifier(),
    'SVC': SVC(),
    'RandomForest': RandomForestClassifier()
}

param_grid = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'DecisionTree': {'max_depth': [3, 5, 10]},
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'RandomForest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]}
}

best_model = None
best_score = 0
best_name = ''

# Perform k-fold cross-validation and hyperparameter tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    clf = GridSearchCV(model, param_grid[name], cv=kf, scoring='accuracy')
    clf.fit(X_train, y_train)
    score = clf.best_score_
    
    if score > best_score:
        best_score = score
        best_model = clf.best_estimator_
        best_name = name

# Train the best model on the full training set and save it
best_model.fit(X_train, y_train)
print(f"Best Model: {best_name} with score: {best_score}")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Test the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
