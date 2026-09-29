from fastapi import FastAPI
from pydantic import BaseModel
import joblib
app = FastAPI()
model = joblib.load("model/iris_model.pkl")
class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "FastAPI ML Inference API"}

@app.post("/predict")
def predict(data: Flower):
    prediction = model.predict([[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]])
    return {"Prediction": int(prediction[0])}