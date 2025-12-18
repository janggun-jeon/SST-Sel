#!/bin/bash

DATASET_TYPE='movie' # choices=['movie', 'music']
LLM_TYPE='gemini-2.0-flash' # choices=['gemini-2.0-flash', 'gpt-4.1-mini', 'gpt-4.1-nano']
API_KEY='your_api_key'

if [ "$DATASET_TYPE" = 'movie' ]; then
    python utils/movielens.py
    python utils/under_sampling.py --dataset_type $DATASET_TYPE
elif [ "$DATASET_TYPE" = 'music' ]; then
    python utils/lastfm.py
fi



python utils/popularity.py --dataset_type $DATASET_TYPE

# SST-Only
python llm_as_recommender.py \
    --dataset_type $DATASET_TYPE \
    --llm_type $LLM_TYPE \
    --api_key $API_KEY

python eval.py \
    --dataset_type $DATASET_TYPE



python utils/thresholding.py --dataset_type $DATASET_TYPE

# SST-Sel
python llm_as_recommender.py \
    --dataset_type $DATASET_TYPE \
    --llm_type $LLM_TYPE \
    --api_key $API_KEY \
    --sst_sel

python eval.py \
    --dataset_type $DATASET_TYPE \
    --sst_sel