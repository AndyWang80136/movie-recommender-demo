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

## Serve Models
```console
# Recall
uvicorn deploy_models.deploy_recall_algorithm:app --port 8000
# Rank
uvicorn deploy_models.deploy_rank_algorithm:app --port 8001
# Rerank
uvicorn deploy_models.deploy_rerank_algorithm:app --port 8002
```

## Run Demo App
```console
streamlit run demo.py
```

## Statistics
### Rank (DCN)
#### Hash Buckets
```
user_id
bucket size: 3000
No training bucket ratio: 0.0397

item_id
bucket size: 2000
No training bucket ratio: 0.0244
```

#### Evaluation Metric
```
Testing AUC: 0.7266
Total test users: 1239
Average user test AUC: 0.6926
Average user test NDCG@8: 0.8096
```

## Citation
MovieLens 1M Dataset
```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
```
