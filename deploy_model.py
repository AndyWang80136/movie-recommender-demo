import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from feature_analysis.data import DatasetLoader
from feature_analysis.process import load_dcn_model
from pydantic import BaseModel

from dataset import *

app = FastAPI()

model = load_dcn_model(ckpt='./output_model/val_best.pth')
model.eval()

dataset_config = joblib.load('./output_model/dataset.pkl')
dataset = DatasetLoader.load(**dataset_config)


class Data(BaseModel):
    data: dict


@app.post("/predict/")
async def predict(input_data: Data):
    test_df = pd.DataFrame(input_data.data)
    trns_df = dataset.transform_df(test_df)
    trns_data = {col: trns_df[col].values for col in trns_df.columns}
    pred_ans = model.predict(trns_data, 128)
    pred_ans = pred_ans.ravel().tolist()
    return pred_ans


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
