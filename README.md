# Movie Recommender Demo

## Installation
1. Install [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) package

2. Type in Terminal
```console
pip install -e .
```

## Dataset
The dataset used in the demo: `MovieLens 1M Dataset`

## Train DCN Model
```console
python train_model.py --config configs/ml-1m.yaml
```

## Serve DCN Model
```console
uvicorn deploy_model:app --port 8000
```

## Run Demo App
```console
streamlit run demo.py
```

## Statistics
### Hash Buckets
```
user_id
bucket size: 3000
No training bucket ratio: 0.0397

item_id
bucket size: 2000
No training bucket ratio: 0.0244
```

## Evaluation Metric
```
Testing AUC: 0.7266
Total test users: 1239
Average user test AUC: 0.6926
Average user test NDCG@8: 0.8096
```

## Demo
**Note: The demo videos were recorded with a smaller window scale**
### Bayesian Recommender
https://github.com/AndyWang80136/movie-recommender-demo/assets/14234143/7bb8d760-c36c-49ce-871b-e313bb2f00a4


Bayesian Recommender uses Bayesian statistics on `movie genres` and `like ratio (created by ratings)` to genreate movie recommendations


### ML1M Recommender
https://github.com/AndyWang80136/movie-recommender-demo/assets/14234143/6f546053-5df0-421d-b129-c225c196c9cc



ML1M recommender offers 3 kinds of algorithms:

- Content-Based Algorithm: Recommendations based on movie genre similarity
- Uesr-Behavior Algorithm: Recommendations based on movie likeness similarity in each user demographics
- Deep Cross Network: Recommendations based on retrieving 50 candidates in total from previous 2 algorithms and ranking by the DCN model 

### DCN Model Test Performance on ML1M
https://github.com/AndyWang80136/movie-recommender-demo/assets/14234143/a93de45c-000f-47c5-bd4a-7bbf035a0673


DCN model test phase performance 

- Show recommendations in test phase
- Inspected NDCG@8 and total test AUC per user

## Citation
MovieLens 1M Dataset
```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
```
