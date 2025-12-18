import warnings;warnings.filterwarnings('ignore')
import argparse
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_type', type=str, default='movie', choices=['movie', 'music'])

config = parser.parse_args()

dataset_name:str
gt_src:str
if config.dataset_type == 'movie':
    dataset_name = 'MovieLens-1M'
else:
    dataset_name = 'LastFm-1K'

with open(os.path.join(os.getcwd(), f'datasets/{dataset_name}/sample.json'), "r", encoding="utf-8") as f:
    sample_dict = json.load(f)
    
popularity: dict[str, int] = {}
for user in sample_dict.values():
    for item in user['items'].values():
        key = item['title']
        val = popularity.get(key, 0) + 1
        popularity[key] = val
        
popular_values = list(popularity.values())
min_popular = min(popular_values)
max_popular = max(popular_values)
popular_range = max_popular - min_popular

normalized_popularity = {key: (val - min_popular) / popular_range for key, val in popularity.items()}

with open(os.path.join(os.getcwd(), 'utils', f'{config.dataset_type}-popularity.json'), "w", encoding="utf-8") as f:
    json.dump(normalized_popularity, f, ensure_ascii=False, indent=2)