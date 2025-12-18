import os
import pandas as pd
import re
import json

files = ['movies', 'ratings', 'users']
prefix = os.path.join(os.getcwd(), 'datasets', 'MovieLens-1M')

for file in files:
    input_file  = os.path.join(prefix, f'{file}.dat')
    output_file = os.path.join(prefix, f'{file}.csv')

    if not os.path.exists(output_file):
        if file == 'movies':
            df = pd.read_csv(input_file, sep="::", engine="python", names=["item_id", "title", "genre"], encoding='latin-1')
            df.to_csv(output_file, index=False)
            print("CSV 파일로 저장 완료:", output_file)
        if file == 'ratings':
            df = pd.read_csv(input_file, sep="::", engine="python", names=["user_id", "item_id", "rating", "timestamp"], encoding='latin-1')
            df.to_csv(output_file, index=False)
            print("CSV 파일로 저장 완료:", output_file)
        if file == 'users':
            df = pd.read_csv(input_file, sep="::", engine="python", names=["user_id", "gender", "age", "job", "postal_code"], encoding='latin-1')
            df.to_csv(output_file, index=False)
            print("CSV 파일로 저장 완료:", output_file)
    else:
        print("이미 CSV 파일이 존재:", output_file)
         
# 파일 경로 설정
users_file = os.path.join(prefix, 'users.csv')
ratings_file = os.path.join(prefix, 'ratings.csv')
movies_file = os.path.join(prefix, 'movies.csv')

# 데이터 읽기
users = pd.read_csv(users_file)
ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

# movies에서 연도 추출 (예: "One Flew Over the Cuckoo's Nest (1975)")
def extract_year(title):
    import re
    m = re.search(r'\((\d{4})\)', title)
    if m:
        return int(m.group(1))
    return None

movies['year'] = movies['title'].apply(extract_year)

movies['genre'] = movies['genre'].apply(lambda x: x.split('|'))

gender_expression = {
    "M": 'male',
    "F": 'female'
}

age_scope = {
    1: "0-18",
    18: "18-24",
    25: "25-34",
	35: "35-44",
	45: "45-49",
	50: "50-55",
	56: "56+"
}

job_number = {
	0:  "unspecified",
	1:  "academic/educator",
	2:  "artist",
	3:  "clerical/admin",
	4:  "college/grad student",
	5:  "customer service",
	6:  "doctor/health care",
	7:  "executive/managerial",
	8:  "farmer",
	9:  "homemaker",
	10:  "K-12 student",
	11:  "lawyer",
	12:  "programmer",
	13:  "retired",
	14:  "sales/marketing",
	15:  "scientist",
	16:  "self-employed",
	17:  "technician/engineer",
	18:  "tradesman/craftsman",
	19:  "unemployed",
	20:  "writer",
}

# movies 딕셔너리 생성
movies_dict = movies.set_index('item_id').to_dict(orient='index')

# 최종 ground-truth 딕셔너리 생성
ground_truth = {}

for _, usr_row in users.iterrows():
    uid = int(usr_row['user_id'])
    gender = str(gender_expression[usr_row['gender']])
    age = str(age_scope[usr_row['age']])
    job = str(job_number[usr_row['job']])
    user_dict = {"gender": gender, "age": age, "job": job, "items":None}
    
    user_ratings = ratings[ratings['user_id'] == uid]
    items = {}
    for _, r in user_ratings.iterrows():
        item_id = int(r['item_id'])
        rating = r['rating']
        timestamp = r['timestamp']
        
        movie_info = movies_dict[item_id]
        item_dict = {
            "rating": int(rating),
            "title": re.sub(r' \(\d{4}\)$', '', movie_info['title']),
            "year": int(movie_info['year']),
            "genre": movie_info['genre'],
            "timestamp": int(timestamp)
        }
        items[item_id] = item_dict
    user_dict["items"] = items
    ground_truth[uid] = user_dict

with open(os.path.join(prefix, 'ground_truth.json'), "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)