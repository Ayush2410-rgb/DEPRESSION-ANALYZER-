import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = 'data/mental_heath_unbanlanced.csv'
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Remove missing values
df = df.dropna(subset=["text", "status"])

# Remove unnecessary ID
df = df.drop(columns=["Unique_ID"], errors="ignore")

print("\nClass Distribution:")
print(df["status"].value_counts())

# ==========================================
# 2. PREPARE DATA
# ==========================================

X = df["text"].astype(str)
y = df["status"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 3. CREATE MODEL
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 5. EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# ==========================================
# 6. SAVE METRICS
# ==========================================

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "dataset_size": len(df)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# ==========================================
# 7. SAVE MODEL
# ==========================================

joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")

# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(
    xticks_rotation=45
)

plt.title("Mental Health Text Classification")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix saved.")

# ==========================================
# 9. CLASS DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

df["status"].value_counts().plot(kind="bar")

plt.title("Dataset Class Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Text Samples")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=300
)

plt.close()

print("Class distribution graph saved.")

print("\n================================")
print("TRAINING COMPLETE")
print("================================")