# A Study on Prompt Selection based on User Preferences in the LLM-as-Recommender System
 This repository was created for LLM as Recommender to experiment with popular item recommendation bias based on prompts.

![Image](https://github.com/user-attachments/assets/aac976a6-5e79-43c6-9be8-ab326f7f973a)

### view: [https://janggun-jeon.github.io/SST-Sel/](https://janggun-jeon.github.io/SST-Sel/)

## Requirements

```shell
  backoff==2.2.1
  matplotlib==3.10.8
  numpy==2.3.5
  openai==2.13.0
  pandas==2.3.3
  protobuf==6.33.2
  PyYAML==6.0.2
  PyYAML==6.0.3
  scikit_learn==1.8.0
```


## Repo Structure

The repository is organized as follows:
- `datasets/`: dataset folder
    - `lastfm-dataset-1K.tar.gz`: [Last-FM 1K dataset](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) compressed file
    - `ml-1m.zip`: [MovieLens 1K dataset](https://grouplens.org/datasets/movielens/1m/) compressed file
- `utils/`: folder for pre-post processing related modules
    - `movielens.py`: MovieLens dataset preprocessing module
    - `lastfm.py`: Last-FM dataset preprocessing module
    - `under_sampling.py`: dataset undersampling module
    - `popularity.py`: item popularity table calculation module
    - `thresholding.py`: popularity threshold calculation module
- `llm_as_recommender.py`: recommending for LLM Model
- `eval.py`: evaluating Rec performance


## Dataset Usage

#### Last-FM 1K datasets
```bash
mv lastfm-dataset-1K.tar.gz ./datasets/
mkdir -p ./LastFm-1K && tar -zxvf lastfm-dataset-1K.tar.gz -C ./LastFm-1k --strip-components=1
```


#### MovieLens 1M datasets
```bash
mv ml-1m.zip ./datasets/
unzip -j ml-1m.zip -d ./MovieLens-1M
```

## Example (Terminal)

```bash
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
```

\* To run the code, you need to have a **Gemini or GPT API key**. 

## Citation
If you use our code, please cite the paper below:
```bibtex
@article{,
  title={A Study on Prompt Selection based on User Preferences in the LLM-as-Recommender System},
  author={Jeon, Janggun and Kim, Namgi},
  journal={},
  year={2025}
}
```
