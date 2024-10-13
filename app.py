from fastapi import FastAPI, Query
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the pre-trained model
model = joblib.load('model.pkl')

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def welcome():
    return {"message": "Welcome to the Diabetes Prediction API"}

@app.post("/predict")
async def predict_post(input: DiabetesInput):
    # Prepare the input for model prediction
    input_data = np.array([[
        input.Pregnancies, input.Glucose, input.BloodPressure, input.SkinThickness,
        input.Insulin, input.BMI, input.DiabetesPedigreeFunction, input.Age
    ]])

    # Use the loaded model for prediction
    result = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if result[0] == 1:
        prediction = "Positive (Diabetic)"
    else:
        prediction = "Negative (Not Diabetic)"

    return {
        "prediction": prediction,
        "probability": float(probability)
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)