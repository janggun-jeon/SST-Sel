import warnings;warnings.filterwarnings('ignore')
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='movie', choices=['movie', 'music'])
config = parser.parse_args()

with open(os.path.join(os.getcwd(), 'sst-history.json'), "r", encoding="utf-8") as f:
    histories = json.load(f)
    
with open(os.path.join(os.getcwd(), 'utils', f'{config.dataset_type}-popularity.json'), "r", encoding="utf-8") as f:
    popularity = json.load(f)

with open(os.path.join(os.getcwd(), 'labels.json'), "r", encoding="utf-8") as f:
    labels = [label for ulabel in json.load(f)['leave-ont-out']['top 20']['NDCG'] for uid, label in ulabel.items()]
    
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

thresholdings = [0, 0.0, 0.0]
m = 0
for itr in range(1, 21):
    m = itr

    scores: list[int] = []
        
    for uid, history in histories.items():
        items = history['items']
        titles = [item['title'] for item in items.values()]
        populars = [popularity[title] for title in titles] 
        populars.sort()
        popular = populars[m-1]
        scores.append(popular)


    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    optimal_threshold:float
    optimal_idx:int

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
   
    if auc > thresholdings[2]:
        thresholdings[0] = m
        thresholdings[1] = float(optimal_threshold)
        thresholdings[2] = auc  

thresholds = {thresholdings[0]: thresholdings[1]}; print(f"🚀 최적 임계값 (Optimal Threshold): {thresholds}")
with open(os.path.join(os.getcwd(), 'utils', 'thresholds.json'), "w", encoding='utf-8') as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=2)