# ML-Fall-25

import math
import pandas as pd
import numpy as np

# 🧩 1. Make your dataset here
# 👉 Each column represents an attribute (feature)
# 👉 The last column is always your target/output (class label)
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Old', 'Middle', 'Young', 'Young', 'Old'],
    'Credit': ['Good', 'Excellent', 'Good', 'Good', 'Excellent', 'Excellent', 'Good', 'Excellent', 'Good', 'Excellent'],
    'Buys': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']  # ← target column
}

# 🧾 2. Convert to a DataFrame (like an Excel table in Python)
df = pd.DataFrame(data)

# ⚙️ 3. Function to calculate entropy (measure of impurity)
def entropy(target_col):
    # Find unique classes and how often they appear (Yes/No counts)
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = 0
    for i in range(len(elements)):
        prob = counts[i] / np.sum(counts)
        entropy_val += -prob * math.log2(prob)   # entropy formula
    return entropy_val

# ⚙️ 4. Function to calculate Information Gain for one column
def info_gain(df, attribute, target_name="Buys"):
    total_entropy = entropy(df[target_name])  # entropy before splitting
    vals, counts = np.unique(df[attribute], return_counts=True)
    weighted_entropy = 0

    # For each unique value in that attribute (like each Age category)
    for i in range(len(vals)):
        subset = df[df[attribute] == vals[i]]
        weighted_entropy += (counts[i]/np.sum(counts)) * entropy(subset[target_name])

    gain = total_entropy - weighted_entropy  # how much entropy decreased
    return gain

# ⚙️ 5. Find which column has the highest info gain (best root node)
gains = {}
for col in df.columns[:-1]:   # skip last column (target)
    gains[col] = info_gain(df, col)

print("Information Gain for each attribute:", gains)
print("🌳 Root Node (Best Attribute):", max(gains, key=gains.get))

print("Information Gain for each attribute:", gains)
print("Root Node (Best Attribute):", max(gains, key=gains.get))
