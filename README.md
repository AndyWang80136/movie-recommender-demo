# Movie Recommender Demo

## Installation
```console
pip install -r requirements/core.txt
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
### Real Life Recommender
https://github.com/AndyWang80136/movie-recommender-demo/assets/14234143/05c5a754-b05f-429c-ba90-2ef7b4471cbe

Real life recommender offers 3 kinds of algorithms:

- Content-Based Algorithm: Recommendations based on movie genre similarity
- Uesr-Behavior Algorithm: Recommendations based on movie likeness similarity in each user demographics
- Deep Cross Network: Recommendations based on retrieving 50 candidates in total from previous 2 algorithms and ranking by the DCN model 

### DCN Model Performance
https://github.com/AndyWang80136/movie-recommender-demo/assets/14234143/32b60127-6e12-4ec2-9950-96cb265eb7d2

DCN model test phase performance 

- Show recommendations in test phase
- Inspected NDCG@8 and total test AUC per user

## Citation
MovieLens 1M Dataset
```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
```
