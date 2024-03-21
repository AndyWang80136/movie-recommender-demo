# Movie Recommender Demo

## Installation
1. Install [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) package

2. Type in Terminal
```console
pip install -e .
```

## Dataset
The dataset used in the demo: `MovieLens 1M Dataset`

## Train DCN Rank Model
```console
python tools/train_rank_model.py --config configs/ml-1m.yaml
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

## Demo
### Movie Recommender
https://github.com/AndyWang80136/movie-recommender-demo/assets/14234143/a112f8d9-a5e1-4f6f-b957-23967b4d385c



## Statistics
### Recall (CF)
#### hyperparameter tuning on val data
```console
python tools/recall_hyperparam_tuning.py --save-dir val_hparam --similarity_top_k 10,20,50,100 --output_top_k 100
```
`val_hparam/best_metrics.json`
```json
{
    "usercf-content": {
        "similarity_top_k": 100,
        "output_top_k": 100,
        "metric_value": 0.26981465977959906
    },
    "usercf-ratings": {
        "similarity_top_k": 100,
        "output_top_k": 100,
        "metric_value": 0.23871668488485528
    },
    "itemcf-ratings": {
        "similarity_top_k": 10,
        "output_top_k": 100,
        "metric_value": 0.21642876390303534
    },
    "itemcf-genres": {
        "similarity_top_k": 10,
        "output_top_k": 100,
        "metric_value": 0.03703561596627432
    }
}
```
#### Evaluation on test data
```console
python tools/recall_evaluate.py --config val_hparam/best_params.json --phase test --save-dir eval_results/
```
`eval_results/metrics.json`
```json
{
    "usercf-content": {
        "recall": 0.23613394701789722
    },
    "usercf-ratings": {
        "recall": 0.24083239836659517
    },
    "itemcf-ratings": {
        "recall": 0.21642782513385445
    },
    "itemcf-genres": {
        "recall": 0.040183930035866816
    }
}
```

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
