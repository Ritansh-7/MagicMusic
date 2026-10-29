from pathlib import Path
import pandas as pd
from recommender import SpotifyRecommendationEngine

data = pd.read_csv(Path(__file__).parent / "spotify_tracks_data.csv")
engine = SpotifyRecommendationEngine(data)
print("Catalog:", engine.catalog_stats())
seed = data.sort_values("popularity", ascending=False).iloc[0]
print(f"\nSeed track: {seed['track_name']} — {seed['artists']}")
recommendations = engine.recommend(str(seed["track_id"]), n=10)
print(recommendations[["track_name", "artists", "track_genre", "popularity", "similarity", "recommendation_score"]].to_string(index=False))
