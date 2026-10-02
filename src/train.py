import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

DATA_PATH = "data/raw/spotify_tracks.csv"

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_log_z"
]

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

df["popularity_class"] = pd.qcut(
    df["popularity"],
    q=3,
    labels=["Low", "Medium", "High"]
)

X = df[FEATURES].copy()
y = df["popularity_class"].copy()

X = X.fillna(X.median())

print("\nPopularity class distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining KNN model with K = 3...")

model = KNeighborsClassifier(
    n_neighbors=3,
    weights="uniform",
    metric="minkowski",
    p=2,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("Generating predictions...")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nFinal KNN Results")
print("-----------------")
print(f"Accuracy:           {accuracy:.6f}")
print(f"Weighted Precision: {precision:.6f}")
print(f"Weighted Recall:    {recall:.6f}")
print(f"Weighted F1-Score:  {f1:.6f}")

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1-Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(
    "outputs/tables/final_evaluation_metrics.csv",
    index=False
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "outputs/tables/final_classification_report.csv"
)

class_order = ["Low", "Medium", "High"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_order
)

pd.DataFrame(
    cm,
    index=class_order,
    columns=class_order
).to_csv(
    "outputs/tables/final_confusion_matrix.csv"
)

pd.DataFrame({
    "Feature": FEATURES
}).to_csv(
    "outputs/tables/model_feature_order.csv",
    index=False
)

joblib.dump(
    model,
    "models/knn_spotify_popularity.pkl"
)

joblib.dump(
    scaler,
    "models/standard_scaler.pkl"
)

print("\nModel saved: models/knn_spotify_popularity.pkl")
print("Scaler saved: models/standard_scaler.pkl")
print("Training pipeline completed successfully.")