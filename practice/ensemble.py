# ===========================================================
# MACHINE LEARNING LAB 05 – ENSEMBLE LEARNING
# ===========================================================
# COURSE CODE: AL3002
# INSTRUCTOR: Alishba Subhani
#
# OBJECTIVE:
# 1. Understand Ensemble Techniques
# 2. Apply Voting Classifier (Hard, Soft, and Weighted)
# 3. Improve Accuracy using Ensemble Methods
# 4. Implement Random Forest, AdaBoost, and XGBoost
# ===========================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ===========================================================
# TASK 1
# ===========================================================
# TASK DESCRIPTION (from lab manual):
# ● Download the dataset
# ● Perform EDA
# ● Check if the dataset is balanced or not (using target variable "Label")
# ● Check for empty records, categorical features, or duplicate records; handle them
# ● Analyze if feature scaling is required
# ● Split dataset: 80% training, 20% testing
# ● From training set, make 70% training and 30% validation
# ● Apply Random Forest, AdaBoost, and XGBoost
# ● Compare training and testing results of all three algorithms
# ===========================================================

# Step 1: Load your dataset
df = pd.read_csv("heart.csv")  # replace with your dataset path

# Step 2: Explore dataset
print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== CHECK MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== CHECK DUPLICATES ===")
print(df.duplicated().sum())

print("\n=== CLASS DISTRIBUTION ===")
print(df['Label'].value_counts())

# Step 3: Split features (X) and target (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Step 4: Feature scaling (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# Step 6: Initialize ensemble models
rf = RandomForestClassifier(random_state=0)
adb = AdaBoostClassifier(random_state=0)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

# Step 7: Train and evaluate
models = {'Random Forest': rf, 'AdaBoost': adb, 'XGBoost': xgb}

for name, model in models.items():
    model.fit(X_train_sub, y_train_sub)
    print(f"\n{name}")
    print(f"Training Accuracy: {model.score(X_train_sub, y_train_sub):.3f}")
    print(f"Testing Accuracy: {model.score(X_test, y_test):.3f}")

# Step 8: Plot comparison
acc_df = pd.DataFrame({
    'Model': ['Random Forest', 'AdaBoost', 'XGBoost'],
    'Train Accuracy': [rf.score(X_train_sub, y_train_sub), adb.score(X_train_sub, y_train_sub), xgb.score(X_train_sub, y_train_sub)],
    'Test Accuracy': [rf.score(X_test, y_test), adb.score(X_test, y_test), xgb.score(X_test, y_test)]
})
sns.barplot(x='Model', y='Test Accuracy', data=acc_df)
plt.title("Task 1: Comparison of Ensemble Methods")
plt.show()


# ===========================================================
# TASK 2
# ===========================================================
# TASK DESCRIPTION (from lab manual):
# ● Use same dataset as in Task 1
# ● Extract only two attributes: restecg and oldpeak
# ● Train a Voting Classifier using Decision Tree, KNN, Random Forest, and XGBoost
# ● Check which voting parameter gives best accuracy (soft or hard)
# ● Check best weights for these models
# ● Plot the Bias and Variance Tradeoff Graph after Voting Classifier
# ===========================================================

# Step 1: Select only two features
X = df[['restecg', 'oldpeak']]
y = df['Label']

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Initialize base models
dt = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=0)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

# Step 4: Hard voting
voting_hard = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)], voting='hard')
voting_hard.fit(X_train, y_train)
print("\nHard Voting Accuracy:", voting_hard.score(X_test, y_test))

# Step 5: Soft voting
voting_soft = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)], voting='soft')
voting_soft.fit(X_train, y_train)
print("Soft Voting Accuracy:", voting_soft.score(X_test, y_test))

# Step 6: Weighted voting
voting_weighted = VotingClassifier(
    estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=[1, 2, 3, 4]
)
voting_weighted.fit(X_train, y_train)
print("Weighted Voting Accuracy:", voting_weighted.score(X_test, y_test))

# Step 7: Plot Bias-Variance tradeoff
train_scores, test_scores, labels = [], [], ['DT', 'KNN', 'RF', 'XGB', 'Voting']

for model in [dt, knn, rf, xgb, voting_soft]:
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.plot(labels, train_scores, marker='o', label='Train Accuracy')
plt.plot(labels, test_scores, marker='o', label='Test Accuracy')
plt.title("Task 2: Bias-Variance Tradeoff after Voting Classifier")
plt.legend()
plt.show()


# ===========================================================
# TASK 3
# ===========================================================
# TASK DESCRIPTION (from lab manual):
# ● Use same dataset as in Task 1
# ● Extract only two attributes: restecg and chol
# ● Train a Voting Classifier using Random Forest and AdaBoost
# ● Plot training and testing accuracy of individual Random Forest and AdaBoost
# ● Plot accuracy graph of Voting Ensemble Technique as well
# ===========================================================

# Step 1: Select features
X = df[['restecg', 'chol']]
y = df['Label']

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Initialize models
rf = RandomForestClassifier(random_state=0)
adb = AdaBoostClassifier(random_state=0)

# Step 4: Train models
rf.fit(X_train, y_train)
adb.fit(X_train, y_train)

# Step 5: Print accuracies
print("\nRandom Forest Accuracy:", rf.score(X_test, y_test))
print("AdaBoost Accuracy:", adb.score(X_test, y_test))

# Step 6: Combine using Voting Classifier
vote = VotingClassifier(estimators=[('rf', rf), ('adb', adb)], voting='soft')
vote.fit(X_train, y_train)
print("Voting Ensemble Accuracy:", vote.score(X_test, y_test))

# Step 7: Plot accuracy comparison
acc_df = pd.DataFrame({
    'Model': ['Random Forest', 'AdaBoost', 'Voting Ensemble'],
    'Accuracy': [rf.score(X_test, y_test), adb.score(X_test, y_test), vote.score(X_test, y_test)]
})
sns.barplot(x='Model', y='Accuracy', data=acc_df)
plt.title("Task 3: Random Forest vs AdaBoost vs Voting Ensemble")
plt.show()

# ===========================================================
# END OF LAB 05 – ENSEMBLE LEARNING
# ===========================================================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Example dataset
df = pd.read_csv("your_dataset.csv")
X = df.drop('target', axis=1)
y = df['target']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict & evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))



from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

df = pd.read_csv("your_dataset.csv")
X = df.drop('target', axis=1)
y = df['target']

# Define model
model = DecisionTreeClassifier()

# Define KFold (e.g., 5 folds)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using cross_val_score
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("K-Fold Accuracies:", scores)
print("Mean Accuracy:", scores.mean())


