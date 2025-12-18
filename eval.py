import warnings;warnings.filterwarnings('ignore')
import argparse
import os
import json
import itertools
import math
import yaml
import pandas as pd
import difflib
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_type', type=str, default='movie', choices=['movie', 'music'])
parser.add_argument('--recommend_num', type=int, default=20)
parser.add_argument('--sst_class', nargs='+', type=str, default=['gender', 'age'])
parser.add_argument('--sst_sel', default=False)

config = parser.parse_args()



prefix = 'sst-sel' if config.sst_sel else 'sst' ; print(f'eval type: {prefix}')
ranking_path = os.path.join(os.getcwd(), f'{prefix}-rankings.csv') 
rankings = pd.read_csv(ranking_path)

positive_path = os.path.join(os.getcwd(), f'{prefix}-positives.json') 
with open(positive_path, "r", encoding="utf-8") as f:
    positives = json.load(f)
    
candidate_path = os.path.join(os.getcwd(), f'{prefix}-candidates.json') 
with open(candidate_path, "r", encoding="utf-8") as f:
    candidates = json.load(f)
    
usr_group_info:dict
if config.dataset_type == 'movie':
    usr_group_info = {'gender': ['male', 'female'], 'age': ['0-18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']}
else:
    usr_group_info = {'gender': ['male', 'female'], 'age': [        '18-24', '25-34'                                  ]}
    


def clean(config, text):
    lines = [line for line in text.splitlines() if line.strip()]

    cleaned = []
    for line in lines:
        splited = line
        
        index = splited.find('**')
        rIndex = splited.rfind('**')
        if rIndex != -1 and index != rIndex:
            splited = splited[:rIndex]
            splited = splited[index + 2:]
            
        index = splited.find('. ')
        if index != -1:
            splited = splited[index + 2:]

        rIndex = splited.rfind(' (')
        if rIndex != -1:
            splited = splited[:rIndex]

        cleaned.append(splited)

    return cleaned[:config.recommend_num]



def create_title_map(rankings, candidates):
    """
    추천 목록에 있는 모든 제목과 실제 제목을 미리 매핑하는 함수
    """
    print("Creating title to actual title map...")
    
    # 1. 추천 결과에 등장하는 모든 영화 제목을 중복 없이 추출
    sens_dict = {}
    neut_dict = {}
    sst_sel_dict = {}
    
    all_titles:list
    if not config.sst_sel:
        sens_list = list(map(lambda x: clean(config, x), rankings['sensitive'].tolist()));print(' precessing sensitive Recs');
        neut_list = list(map(lambda x: clean(config, x), rankings['neutral'].tolist()));print(' processing neutral Recs');
    
        for uid, sens_rank, neut_rank in zip(rankings['uid'], sens_list, neut_list):
            sens_dict[uid] = sens_rank
            neut_dict[uid] = neut_rank
    
        all_titles = list(set(itertools.chain.from_iterable(itertools.chain(sens_list, neut_list))))
    else:
        sst_sel_list = list(map(lambda x: clean(config, x), rankings['sst_sel'].tolist()));print(' processing SST-Sel Recs');
        
        for uid, sst_sel_rank in zip(rankings['uid'], sst_sel_list):
            sst_sel_dict[uid] = sst_sel_rank
            
        all_titles = list(set(itertools.chain.from_iterable(sst_sel_list)))
        
    # 2. 실제 DB에 있는 제목 리스트
    db = list(itertools.chain.from_iterable(candidates.values()))
    db_titles = [titles.lower() for titles in db]
    
    # 3. RapidFuzz를 사용해 한 번에 매핑 수행
    title_map = {}
    # process.extractOne은 각 쿼리에 대해 가장 유사한 하나를 찾아줍니다.
    # score_cutoff은 유사도 점수 임계값입니다.
    itr = 1
    for title in all_titles:
        print(f"\r[{itr:04d}/{len(all_titles)}] title matching...", end='', flush=True);itr += 1
        match = difflib.get_close_matches(title.lower(), db_titles, n=1, cutoff=0.666)
        if match:
            # match 결과: ('찾은 제목', 점수, 인덱스)
            title_map[title] = db[db_titles.index(match[0])]  
        else:
            
            title_map[title] = title # 매칭 실패 시 원래 제목 사용        
    
    print()
    if not config.sst_sel:
        return title_map, sens_dict, neut_dict
    else:
        return title_map, sst_sel_dict, None



# 메인 로직 시작 전에 title_map을 미리 생성합니다.
title_map:dict
sens_dict:dict
neut_dict:dict
sst_sel_dict:dict
if not config.sst_sel:
    title_map, sens_dict, neut_dict = create_title_map(rankings, candidates)
    sst_sel_dict = None
else:
    title_map, sst_sel_dict, _ = create_title_map(rankings, candidates)
    sens_dict = None
    neut_dict = None    



def ndcg(p, recs, aspect='llo'):
    if aspect=='llo':
        if p not in recs:
            return 0.0
        else:
            return 1.0 / math.log2(recs.index(p) + 2)



def hr(p, recs, aspect='llo'):
    if aspect=='llo':
        if p not in recs:
            return 0.0
        else:
            return 1.0



def top_k_listing(rec, title_map, top_k):
    return [title_map.get(title, title) for title in rec[:top_k]]



def calc_metric_at_k(positive, title_map, *dicts, aspect='llo', metric='ndcg', top_k=20):
    p = positive[1]['title']
    
    metric_k:dict
    if not config.sst_sel:
        [sens_recs, neut_recs] = dicts
        s = top_k_listing(sens_recs, title_map, top_k)
        n = top_k_listing(neut_recs, title_map, top_k)
        
        label:int
        metric_k = {'sensitive': None, 'neutral': None}
        
        if metric == 'ndcg':    
            metric_k['sensitive'] = ndcg(p, s, aspect)
            metric_k['neutral']   = ndcg(p, n, aspect)
    
            label =  1 if metric_k['sensitive'] >= metric_k['neutral'] else 0     
        elif metric == 'hr':
            metric_k['sensitive'] = hr(p, s, aspect)
            metric_k['neutral']   = hr(p, n, aspect)
    
            label =  1 if metric_k['sensitive'] >= metric_k['neutral'] else 0
    
        return metric_k, label
    else:
        [sst_sel_recs] = dicts
        b = top_k_listing(sst_sel_recs, title_map, top_k)
        
        metric_k = {'sst-sel': None}
        
        if metric == 'ndcg': 
            metric_k['sst-sel'] = ndcg(p, b, aspect)
        elif metric == 'hr':
            metric_k['sst-sel'] = hr(p, b, aspect)
        
        return metric_k



def get_metric_with_rec(positives, title_map, *dicts, aspect='llo', metric='ndcg', top_k=20):   
    metric_dict:dict
    itr = 1
    metrics_k2 = {}
    if not config.sst_sel:
        metric_dict = {'average': [], 'labels': []}
        [sens_recs, neut_recs, _] = dicts
        for sens_recs, neut_recs, uid, positive in zip(sens_dict.values(), neut_dict.values(), positives.keys() ,positives.values()):
            print(f"\r[{itr:04d}/{len(positives):04d}]) top {top_k} evaluation..", end='', flush=True); itr += 1; 
            metrics_k, label = calc_metric_at_k(positive['items'], title_map, sens_recs, neut_recs, aspect=aspect, top_k=top_k, metric=metric)
            metric_dict['average'].append(metrics_k);metrics_k2[uid] = metrics_k
            metric_dict['labels'].append({uid: label})
            for sst in config.sst_class:
                sst_attr = positive[sst]
                if sst_attr not in metric_dict:
                    metric_dict[sst_attr] = []
                metric_dict[sst_attr].append(metrics_k) 
    else:
        metric_dict = {'average': []}
        [_, _, sst_sel_dict] = dicts
        for sst_sel_recs, uid, positive in zip(sst_sel_dict.values(), positives.keys(), positives.values()):
            print(f"\r[{itr:04d}/{len(positives):04d}] top {top_k} evaluation...", end='', flush=True); itr += 1; 
            metrics_k = calc_metric_at_k(positive['items'], title_map, sst_sel_recs, aspect=aspect, top_k=top_k, metric=metric)
            metric_dict['average'].append(metrics_k);metrics_k2[uid] = metrics_k
            for sst in config.sst_class:
                sst_attr = positive[sst]
                if sst_attr not in metric_dict:
                    metric_dict[sst_attr] = []
                metric_dict[sst_attr].append(metrics_k) 
    return metric_dict, metrics_k2



def calc_mean_attr(metric_dict):
    if not config.sst_sel:
        metric_s_mean, metric_n_mean = [], []
        for metrics in metric_dict:
            metric_s, metric_n = metrics.values()
            metric_s_mean.append(metric_s)
            metric_n_mean.append(metric_n)
            
        return {'sensitive': np.mean(metric_s_mean).item(), 'neutral': np.mean(metric_n_mean).item()}
    else:
        metric_b_mean = []
        for metrics in metric_dict:
            metric_b = next(iter(metrics.values()))
            metric_b_mean.append(metric_b)
            
        return {'sst-sel': np.mean(metric_b_mean).item()}
    
def calc_mean_metric_k(metric_dict, metrics_k):
    metric_dict['average'] = calc_mean_attr(metric_dict['average'])

    sst = usr_group_info.values()
    for sst_attrs in sst:
        for attr in sst_attrs:
            metric_dict[attr] = calc_mean_attr(metric_dict[attr])

    return metric_dict, metrics_k



def return_eval(positives, title_map, *dicts, aspect, metric, top_k): 
    metric_dict, metrics_k = calc_mean_metric_k(*get_metric_with_rec(positives, title_map, *dicts, aspect=aspect, top_k=top_k, metric=metric))
    
    if not config.sst_sel:
        labels = metric_dict.pop('labels')
        
        return metric_dict, labels, metrics_k
    else:   
        return metric_dict, None, metrics_k



namespace = {
    'aspect': {
        'leave-ont-out': 'llo'
    },
    'top_k': {
        # 'top 3' : 3,
        'top 5': 5,
        # 'top 10' : 10,
        # 'top 12': 12,
        # 'top 15' : 15,
        'top 20': 20,
    },
    'metric': {
        'NDCG': 'ndcg',
        'Hit-Rate': 'hr',
    },
}

eval_dict = {}
labels_dict = {}
metrics_k2 = {'NDCG': [], 'Hit-Rate': []}
for aspect in namespace['aspect'].keys(): # [true preference alignmnet, ground-truth similarity] aspect compare-target selection
    eval_dict[aspect] = {}
    labels_dict[aspect] = {}
    for top_k in namespace['top_k'].keys():
        eval_dict[aspect][top_k] = {}
        labels_dict[aspect][top_k] = {}
        for metric in namespace['metric']:
            eval_dict[aspect][top_k][metric] = {}
            results, labels, metrics_k = return_eval(positives, title_map, sens_dict, neut_dict, sst_sel_dict, aspect=namespace['aspect'][aspect], top_k=namespace['top_k'][top_k], metric=namespace['metric'][metric])
            for group, comp in results.items():
                eval_dict[aspect][top_k][metric][group] = {}
                for sst_key, value in comp.items():
                    eval_dict[aspect][top_k][metric][group][sst_key] = value
            metrics_k2[metric].append(metrics_k)
            if not config.sst_sel:
                labels_dict[aspect][top_k][metric] = labels

print()
if not config.sst_sel:
    with open(os.path.join(os.getcwd(), f'{prefix}-results.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(eval_dict, f, allow_unicode=True, sort_keys=False)
        
    with open(os.path.join(os.getcwd(), 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels_dict, f, indent=4)
else:
    with open(os.path.join(os.getcwd(), f'{prefix}-results.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(eval_dict, f, allow_unicode=True, sort_keys=False)