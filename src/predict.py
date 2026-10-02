import joblib
import pandas as pd

MODEL_PATH = "models/knn_spotify_popularity.pkl"
SCALER_PATH = "models/standard_scaler.pkl"
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

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(DATA_PATH)

sample = df[FEATURES].iloc[[0]].copy()

sample = sample.fillna(
    df[FEATURES].median()
)

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
probabilities = model.predict_proba(sample_scaled)[0]

print("Spotify Track Popularity Prediction")
print("------------------------------------")

print("\nInput Features:")

for feature in FEATURES:
    print(f"{feature}: {sample.iloc[0][feature]:.6f}")

print("\nPredicted Popularity Class:")
print(prediction)

print("\nNeighbor Vote Probabilities:")

for class_name, probability in zip(
    model.classes_,
    probabilities
):
    print(f"{class_name}: {probability:.6f}")

distances, indices = model.kneighbors(sample_scaled)

print("\nNearest Neighbors:")

for rank, (distance, index) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):
    print(
        f"Neighbor {rank} | "
        f"Training Index: {index} | "
        f"Distance: {distance:.6f}"
    )