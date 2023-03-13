from util import *
from fastapi import FastAPI, UploadFile, File
import logging

logger = logging.getLogger("uvicorn")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

app = FastAPI()

@app.get("/")
async def main():
    return {
        "message": "Hello, World"
    }


@app.post("/predict")
async def predict(file: UploadFile, period: int):
    data = await load_data(file.file)
    #print(len(data))
    data = await preprocess(data)
    features = await extract_features(data, period)
    model = await classify(features)
    predictions = await fit_and_predict(model, data, period)
    # print(model)
    # print(predictions)

    return {
        "predictedModel": model,
        "predictions" : predictions
    }
