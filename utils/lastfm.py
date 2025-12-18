import os
import pandas as pd
import json

prefix = os.path.join(os.getcwd(), 'datasets', 'LastFm-1K')
         
# 파일 경로 설정
users_file = os.path.join(prefix, 'userid-profile.tsv')
ratings_file = os.path.join(prefix, 'userid-timestamp-artid-artname-traid-traname.tsv')

# 데이터 읽기
users = pd.read_csv(users_file, sep='\t')
ratings = pd.read_csv(ratings_file, sep='\t', header=None, names=['userid', 'timestamp', 'musicbrainz-artist-id', 'artist-name', 'musicbrainz-track-id', 'track-name'], on_bad_lines='skip')

clean_users = users.dropna(subset=['gender']).dropna(subset=['age']).drop(['country', 'registered'], axis=1)

bins = [0, 18, 25, 35, 45, 50, 56, 150]
labels = ['0-18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']

clean_users = users.dropna(subset=['gender']).dropna(subset=['age']).drop(['country', 'registered'], axis=1)
clean_users['age'] = pd.cut(clean_users['age'], bins=bins, labels=labels, right=False)

male_users = clean_users[clean_users['gender'] == 'm']
female_users = clean_users[clean_users['gender'] == 'f']

male_users = male_users.copy()
male_users['gender'] = 'male'

female_users = female_users.copy()
female_users['gender'] = 'female'

sample_idx = [[], []]
for age in set(male_users['age']) & set(female_users['age']):
    male_age = male_users[male_users['age'] == age]
    female_age = female_users[female_users['age'] == age]
    
    n = min(len(male_age), len(female_age))
    if n >= 5:
        sample_idx[0].extend(male_age.sample(n, random_state=42).index)
        sample_idx[1].extend(female_age.sample(n, random_state=42).index)

male_users = male_users.loc[sample_idx[0]]
female_users = female_users.loc[sample_idx[1]]

sample_users = pd.concat([male_users, female_users], ignore_index=True)
sample_users['age'] = sample_users['age'].cat.remove_unused_categories()
# sample_users['age'].value_counts()

clean_ratings = ratings[ratings['userid'].isin(sample_users['#id'])].dropna(subset=['musicbrainz-artist-id']).dropna(subset=['musicbrainz-track-id']).drop(['musicbrainz-artist-id'], axis=1)
# clean_ratings.head()

sample_users = sample_users.copy()
sample_users['#id'] = sample_users['#id'].str.split('_').str[1].astype(int)
# sample_users.head()

pattern = r'[^\x20-\x7E]'
clean_ratings = clean_ratings[~clean_ratings['artist-name'].str.contains(pattern, regex=True)]

clean_ratings['timestamp'] = pd.to_datetime(clean_ratings['timestamp'])
clean_ratings = clean_ratings.sort_values(by='timestamp', ascending=True)
clean_ratings = clean_ratings.drop_duplicates(subset=['userid', 'artist-name', 'track-name'], keep='first')

rating_counts = clean_ratings['userid'].value_counts()
sample_idx = rating_counts[rating_counts > 20].index # .intersection(rating_counts[rating_counts < 999999].index) 
sample_ratings = clean_ratings[clean_ratings['userid'].isin(sample_idx)]

sample_ratings = sample_ratings.copy()
sample_ratings['userid'] = sample_ratings['userid'].str.split('_').str[1].astype(int)

sample_ratings['trackid'], _ = pd.factorize(sample_ratings['musicbrainz-track-id'])
sample_ratings = sample_ratings.drop(['musicbrainz-track-id'], axis=1)

sample_ratings.head(-10)

import pandas as pd
import json

# (이미지상 sample_users의 컬럼이 #id로 되어 있어 수정 필요)
if '#id' in sample_users.columns:
    sample_users = sample_users.rename(columns={'#id': 'userid'})

sample_users['userid'] = sample_users['userid'].astype(int)
sample_ratings['userid'] = sample_ratings['userid'].astype(int)
sample_ratings['trackid'] = sample_ratings['trackid'].astype(int)

# 이미 datetime 객체라면 바로 변환, 문자열이라면 to_datetime 후 변환
if not pd.api.types.is_datetime64_any_dtype(sample_ratings['timestamp']):
    sample_ratings['timestamp'] = pd.to_datetime(sample_ratings['timestamp'])

# 나노초 단위를 초 단위로 변경 (int64)
sample_ratings['timestamp_int'] = sample_ratings['timestamp'].astype('int64') // 10**9

# inner join: 두 데이터셋에 모두 존재하는 유저만 남깁니다.
merged_df = pd.merge(sample_ratings, sample_users[['userid', 'gender', 'age']], on='userid', how='inner')

music_database = {}

# uid: 유저 아이디, group: 그 유저의 모든 청취 기록 데이터프레임
for uid, group in merged_df.groupby('userid'):
    group = group.sort_values(by='timestamp_int', ascending=False)
    group = group.iloc[:21]
    
    
    # 유저 메타 정보 추출
    user_gender = group['gender'].iloc[0]
    user_age = str(group['age'].iloc[0]) # 문자열로 확실히 변환 ('18-24')
    
    # items 딕셔너리 생성
    items_dict = {}
    
    # 해당 유저의 청취 기록을 하나씩 순회하며 딕셔너리에 담기
    for _, row in group.iterrows():
        tid = int(row['trackid']) # 키는 정수형
        
        items_dict[tid] = {
            "title": row['track-name'],
            "artist": row['artist-name'],
            "timestamp": int(row['timestamp_int'])
        }
    
    # 최종 딕셔너리에 유저 추가
    music_database[int(uid)] = {
        "gender": user_gender,
        "age": user_age,
        "items": items_dict
    }

# 파일로 저장 (.json)
sample_drc = os.path.join(os.getcwd(), 'datasets', 'LastFm-1K', 'sample.json')
with open(sample_drc, 'w', encoding='utf-8') as f:
    json.dump(music_database, f, indent=4, ensure_ascii=False)

print(f"총 {len(music_database)}명의 유저 데이터가 변환되었습니다.")