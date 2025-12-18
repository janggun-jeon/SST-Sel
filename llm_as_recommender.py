import warnings;warnings.filterwarnings('ignore')
import argparse
import os
import json
import backoff

import openai
import google.generativeai as genai
import random
import pandas as pd

class RateLimitErrorException(openai.RateLimitError, Exception): pass
class APIErrorException(openai.APIError, Exception): pass
class APIConnectionErrorException(openai.APIConnectionError, Exception): pass
class TimeoutException(openai.Timeout, Exception): pass

openai.RateLimitError = RateLimitErrorException
openai.APIError = APIErrorException
openai.APIConnectionError = APIConnectionErrorException
openai.Timeout = TimeoutException

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_type', type=str, default='movie', choices=['movie', 'music'])
parser.add_argument('--split_type', type=str, default='leave-one-out')
parser.add_argument('--profile_num', type=int, default=20)
parser.add_argument('--recommend_num', type=int, default=20)
parser.add_argument('--sst_class', nargs='+', type=str, default=['gender', 'age'])
parser.add_argument('--sst_sel', default=False, action='store_true')
parser.add_argument('--llm_type', type=str, default='gemini-2.0-flash', choices=['gemini-2.0-flash', 'gpt-4.1-mini', 'gpt-4.1-nano'])
parser.add_argument('--api_key', type=str, default='AIzaSyD6OZhInYkUwKqh1bwP-6AsTenFKhhL5Kk')

config = parser.parse_args()



sample_src:str
if config.dataset_type == 'movie':
    sample_src = os.path.join(os.getcwd(), 'datasets', 'MovieLens-1M', 'sample.json')
else:
    sample_src = os.path.join(os.getcwd(), 'datasets', 'LastFm-1K', 'sample.json')

sample_dict:dict  
with open(sample_src, "r", encoding="utf-8") as f:
    sample_dict = json.load(f)

popularity:dict
with open(os.path.join(os.getcwd(),'utils', f'{config.dataset_type}-popularity.json'), "r", encoding="utf-8") as f:
    popularity = json.load(f)

thresholds:dict
if config.sst_sel:
    with open(os.path.join(os.getcwd(), 'utils', 'thresholds.json')   , "r", encoding="utf-8") as f:
        thresholds = json.load(f)

if 'gemini' in config.llm_type:
    genai.configure(api_key=config.api_key)
    print(f'LLM-as-Recommender: {config.llm_type}')
elif 'gpt' in config.llm_type:
    openai.api_key = config.api_key
    print(f'LLM-as-Recommender: {config.llm_type}')


        
def sampling(sample_dict, split_type=config.split_type, profile_num=config.profile_num, recommend_num=config.recommend_num):
    histories = {}
    positives = {}
    candidates = {}
    
    all_titles: list[str] = []
    for items in sample_dict.values():
        for item in items['items'].values():
            title = item['title']
            all_titles.append(title)
              
    for uid, usr in sample_dict.items():
        item_list = []
        
        if split_type == 'leave-one-out':
            unsorted = list(usr['items'].items())
            item_list = sorted(unsorted, key=lambda x: x[1]['timestamp'])

        profile_items = dict(item_list[-(profile_num+1):-1])
        positive_item = item_list[-1]
        
        interaction_titles = [item['title'] for _, item in item_list]
        negative_titles = random.sample(list(set(all_titles) - set(interaction_titles)), recommend_num-1)
        candidate_list = negative_titles + [positive_item[1]['title']]
        random.shuffle(candidate_list)
        
        histories[uid] = {
            'gender': usr['gender'],
            'age': usr['age'],
            'items': profile_items
        }
        positives[uid] = {
            'gender': usr['gender'],
            'age': usr['age'],
            'items': positive_item
        }
        candidates[uid] = candidate_list

    prefix = os.path.join(os.getcwd(), 'sst-sel' if config.sst_sel else 'sst')
    with open(f'{prefix}-history.json', "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    
    with open(f'{prefix}-positives.json', "w", encoding="utf-8") as f:
        json.dump(positives, f, ensure_ascii=False, indent=2)
        
    with open(f'{prefix}-candidates.json', "w", encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        
    return histories, candidates



def profiling(history):
    profiles = []
    for _, item in history.items():
        title = item['title']
        
        if config.dataset_type == 'movie':
            # genre = '|'.join(item['genre'])
            year = item['year']
            rating = item['rating']
            profile = f"\'{title}\' (Year: {year}, Rating: {rating}/5)"
        elif config.dataset_type == 'music':
            artist = item['artist']
            profile = f"\'{title}\' - ( Artist: {artist} )"
            
        profiles.append(profile)
    
    return ', '.join(profiles)
    
    
    
def prompting(history, sst, candidates):
    sensitive = []
    neutral = []

    profiles = profiling(history)
    
    if config.dataset_type == 'movie':
        formats = "**Movie_Title (release year)**"
        content_1 = 'movie'
        content_2 = 'film'
        action = 'watch'
        info = 'release year'
    elif config.dataset_type == 'music':
        formats = "'Music_Track_Title' - ( artist_name )"
        content_1 = 'music'
        content_2 = 'track'
        action = 'stream'
        info = 'artist name'
    
    if 'gpt' in config.llm_type:
        system_msg = f"""You are a movie recommendation system

System Response Format:
1. {formats}\n\n2. {formats}\n\n...\n\nN. {formats}"""

        sensitive.append({"role": "system","content": system_msg})
        neutral.append({"role": "system","content": system_msg})

        sPrompt = f"""User Request:
The user, identified as {sst}, has previously {action}ed the following {content_1} {content_2}s in the order

{profiles}

Taking into account the user's sensitivity ({sst}), please rank these {config.recommend_num} {content_1} below by measuring the possibilities that the user would like to {action}ing next \
most, according to the given viewing history above.
Now there are {config.recommend_num} candidate {content_2}s that the user can consider to view next

{' | '.join(candidates)}

The system MUST rank the given candidate {content_1}s.
Your response MUST ONLY be the ranked list of {content_2} titles and their {info}s in the specified format and nothing else.
DO NOT include any explanations, analysis, introductions, summaries, or any other text.
The system can not recommend {content_1}s that are not in the given candidate list."""
        
        nPrompt = f"""User Request:
The user has previously {action}ed the following {content_1} {content_2}s in the order

{profiles}

Please rank these {config.recommend_num} {content_1}s below by measuring the possibilities that the user would like to {action}ing next \
most, according to the given viewing history above.
Now there are {config.recommend_num} candidate {content_2}s that the user can consider to view next

{' | '.join(candidates)}

The system MUST rank the given candidate {content_1}s.
Your response MUST ONLY be the ranked list of {content_2} titles and their {info}s in the specified format and nothing else.
DO NOT include any explanations, analysis, introductions, summaries, or any other text.
The system can not recommend {content_1}s that are not in the given candidate list."""

        sensitive.append({"role":"user", "content": sPrompt})
        neutral.append({"role":"user", "content": nPrompt})
        
    elif 'gemini' in config.llm_type:
        sPrompt = f"""System Role: movie recommendation system

System Response Format:
1. {formats}\n\n2. {formats}\n\n...\n\nN. {formats}
    
    
User Request:
The user, identified as {sst}, has previously {action}ed the following {content_1} {content_2}s in the order

{profiles}

Taking into account the user's sensitivity ({sst}), please rank these {config.recommend_num} {content_1}s below by measuring the possibilities that the user would like to {action}ing next \
most, according to the given viewing history above.
Now there are {config.recommend_num} candidate {content_2}s that the user can consider to view next

{' | '.join(candidates)}

The system MUST rank the given candidate {content_1}s.
Your response MUST ONLY be the ranked list of {content_2} titles and their {info}s in the specified format and nothing else.
DO NOT include any explanations, analysis, introductions, summaries, or any other text.
The system can not recommend {content_1}s that are not in the given candidate list."""
        sensitive.append(sPrompt)

        nPrompt = f"""System Role: movie recommendation system

    System Response Format:
    1. {formats}\n\n2. {formats}\n\n...\n\nN. {formats}
        
        
    User Request:
    The user has previously {action}ed the following {content_1} {content_2}s in the order

    {profiles}

    Please rank these {config.recommend_num} {content_1}s below by measuring the possibilities that the user would like to {action}ing next \
    most, according to the given viewing history above.
    Now there are {config.recommend_num} candidate {content_2}s that the user can consider to view next

    {' | '.join(candidates)}

    The system MUST rank the given candidate {content_1}s.
    Your response MUST ONLY be the ranked list of {content_2} titles and their {info}s in the specified format and nothing else.
    DO NOT include any explanations, analysis, introductions, summaries, or any other text.
    The system can not recommend {content_1}s that are not in the given candidate list."""
        neutral.append(nPrompt)
       
    return sensitive, neutral



def is_bias(histories, popularity, ratio, threshold):
    scores = []
    for item in histories.values():
        title = item['title']
        scores.append(popularity[title])
        
    scores.sort()
    
    return False if scores[ratio-1] <= threshold else True



model:None
Params:dict
if 'gemini' in config.llm_type:
    model = genai.GenerativeModel(config.llm_type)
    
    Params = {
        'temperature':0.6,
        'top_p':0.1, 
    }
    
    
    
def call_gemini(prompt):
    response = model.generate_content(
        prompt,
        generation_config=Params
    )
    return response._result.candidates[0].content.parts[0].text 



@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.Timeout), max_time=60)
def request_post(**kwargs):
    response = openai.chat.completions.create(**kwargs)
    return response

if 'gpt' in config.llm_type:
    Params = {
        "model": config.llm_type, #"gpt-3.5-turbo",
        'messages': [],
        'n':1,
        'temperature':0.6, # 'temperature':0.6,
        'top_p':0.1, # 'top_p':0.1,
        'frequency_penalty':0,
        'presence_penalty':0
    }    



def call_gpt(params, prompt):    
    params['messages'] = prompt
    response = request_post(**params)
    return response.choices[0].message.content



def get_space_buffer(space_length, reply):
    space_length = max(space_length, len(reply))
    buffer = ' ' * (space_length - len(reply))
    return space_length, buffer
    
    
    
results = []
histories, candidates = sampling(sample_dict, split_type=config.split_type, profile_num=config.profile_num)

space_length = 0
for idx, uid in enumerate(histories.keys()):
    sst = [histories[uid][sst] for sst in config.sst_class]
    sensitive, neutral = prompting(histories[uid]['items'], ' '.join(sst), candidates[uid])
    
    Reply:str
    usr_bias:bool
    if config.sst_sel:
        usr_bias = is_bias(histories[uid]['items'], popularity, int(next(iter(thresholds.keys()))), next(iter(thresholds.values())))
        prompt = sensitive if usr_bias else neutral
        
        if 'gemini' in config.llm_type:
            Reply = call_gemini(prompt)
        elif 'gpt' in config.llm_type:
            Reply = call_gpt(Params, prompt)
            
        results.append({'usr_bias':{'biased' if usr_bias else 'unbiased'}, 'top_k': config.recommend_num, 'uid': uid,'sst-sel': Reply})         
    else:   
        sReply:str
        nReply:str
        if 'gemini' in config.llm_type:          
            sReply = call_gemini(sensitive) 
            nReply = call_gemini(neutral)   
        elif 'gpt' in config.llm_type:
            sReply = call_gpt(Params, sensitive)
            nReply = call_gpt(Params, sensitive)

            
        Reply = sReply
         
        results.append({'usr_bias':'not considered', 'top_k': config.recommend_num, 'uid': uid, 'sensitive': sReply, 'neutral': nReply}) 
    
    space_buffer = '          '
    try:
        reply = Reply.split('\n')[0].split('\'')[1]
        space_length, space_buffer = get_space_buffer(space_length, reply)
        if config.sst_sel:
            print(f"\r[{idx:04d}/{len(histories.keys())}] Usr Bias: {'biased' if usr_bias else 'unbiased'}, top_k: {config.recommend_num}, uid: {uid}, content: {reply}{space_buffer}  ", end='', flush=True)
        else:
            print(f"\r[{idx:04d}/{len(histories.keys())}] Usr Bias: {'not considered'}, top_k: {config.recommend_num}, uid: {uid}, content: {reply}{space_buffer}", end='', flush=True)

    except IndexError as e:
        if config.sst_sel:
            print(f"\r[{idx:04d}/{len(histories.keys())}] Usr Bias: {'biased' if usr_bias else 'unbiased'}, top_k: {config.recommend_num}, uid: {uid}, content: excessive string breaking{space_buffer}  ", end='', flush=True)
        else:
            print(f"\r[{idx:04d}/{len(histories.keys())}] Usr Bias: {'not considered'}, top_k: {config.recommend_num}, uid: {uid}, content: excessive string breaking{space_buffer}", end='', flush=True)

 
prefix = os.path.join(os.getcwd(), 'sst-sel' if config.sst_sel else 'sst')
df = pd.DataFrame(results)
df.to_csv(f'{prefix}-rankings.csv', index=False, encoding='utf-8')
print(f"\n{'SST-Sel' if config.sst_sel else 'SST-Only'} Recs end.")