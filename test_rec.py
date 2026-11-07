from tests.test_recommender import sample_data
from recommender import SpotifyRecommendationEngine

engine = SpotifyRecommendationEngine(sample_data())
result = engine.recommend('a', n=3)
print('Recommendations for seed song A:')
for i, track in enumerate(result.itertuples(), 1):
    print(f'{i}. {track.track_name} by {track.artists}')
print(f'Total: {len(result)} songs')