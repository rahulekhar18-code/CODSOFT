import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset
df = pd.read_csv("iris.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nSpecies Count:")
print(df["species"].value_counts())

# -----------------------------
# Species Distribution Graph
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(x="species", data=df)
plt.title("Species Distribution")
plt.savefig("species_distribution.png")
plt.show()

# -----------------------------
# Pair Plot Graph
# -----------------------------
pair_plot = sns.pairplot(df, hue="species")
pair_plot.savefig("pair_plot.png")
plt.show()

# -----------------------------
# Features and Target
# -----------------------------
X = df[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

y = df["species"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# Sample Prediction
# -----------------------------
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

prediction = model.predict(sample)

print("\nSample Flower Prediction:")
print(prediction[0])

print("\nImages Saved Successfully:")
print("species_distribution.png")
print("pair_plot.png")
print("confusion_matrix.png")