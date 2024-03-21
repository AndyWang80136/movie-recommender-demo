import joblib
import pandas as pd
from fastapi import FastAPI
from feature_analysis.data import DatasetLoader
from feature_analysis.process import load_dcn_model
from pydantic import BaseModel

from movie_recommender.data.dataset import *

app = FastAPI()

model = load_dcn_model(ckpt='./output_model/val_best.pth')
model.eval()

dataset_config = joblib.load('./output_model/dataset.pkl')
dataset = DatasetLoader.load(**dataset_config)


class Data(BaseModel):
    df: dict


@app.post("/rank/dcn")
async def dcn_ranker(data: Data):
    test_df = pd.DataFrame(data.df)
    trns_df = dataset.transform_df(test_df)
    trns_data = {col: trns_df[col].values for col in trns_df.columns}
    pred_ans = model.predict(trns_data, 128)
    pred_ans = pred_ans.ravel().tolist()
    test_df['score'] = pred_ans
    return test_df[['user_id', 'item_id', 'score']].to_json(orient='split')
