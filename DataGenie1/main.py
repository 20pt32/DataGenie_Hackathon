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
    data = await preprocess_predict(data)

    return {
        "predictions" : preprocess_predict
    }
