import warnings;warnings.filterwarnings('ignore')
import argparse
import os
import json

import random
import itertools

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_type', type=str, default='movie', choices=['movie', 'music'])
parser.add_argument('--profile_num', type=int, default=20)

config = parser.parse_args()

usr_group_info:dict
if config.dataset_type == 'movie':
    usr_group_info = {'gender': ['male', 'female'], 'age': ['0-18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']}
else:
    usr_group_info = {'gender': ['male', 'female'], 'age': [        '18-24', '25-34'                                  ]}

def group_sampling(GT, uids, sst):
    group = {}
    for attr in usr_group_info[sst]:
        group[attr] = []

    for uid in uids:
        attr = GT[uid][sst]
        group[attr].append(uid)
    
    return group.values()

if config.dataset_type == 'movie':
    sample_src = os.path.join(os.getcwd(), 'datasets', 'MovieLens-1M', 'sample.json') #'datasets', 'MovieLens-1M', 'sample.json')
    gt_src = os.path.join(os.getcwd(), 'datasets', 'MovieLens-1M', 'ground_truth.json')
    
    if not os.path.exists(sample_src):
        with open(gt_src, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        
        users = {uid: usr for uid, usr in ground_truth.items() if len(usr['items']) >= (config.profile_num + 1)}

        limit_num = []
        gender_age_group = []
        gender_group = group_sampling(ground_truth, users.keys(), sst='gender')

        for gender in gender_group:
            age_group = group_sampling(ground_truth, gender, sst='age')
            gender_age_group.append(age_group)
            limit_num.append(min(len(age) for age in age_group))
        limit_num = min(limit_num)

        sample_ids = []
        for gender_age in gender_age_group:
            for age in gender_age:
                sample_ids.append(random.sample(age, min(len(age), 3 * limit_num)))
        sample_ids = list(itertools.chain(*sample_ids))
        random.shuffle(sample_ids)

        sample_dict = {uid: users[uid] for uid in sample_ids}
        with open(sample_src, "w", encoding="utf-8") as f:
            json.dump(sample_dict, f, ensure_ascii=False, indent=2)    