from fastapi import FastAPI, Query, HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd



encoder = joblib.load('./transform_encode.joblib')
model = joblib.load('./best_rf_mobel.joblib')

app = FastAPI()


class features(BaseModel):
    tenure:str
    montant: float
    frequence_rech: float
    revenue: float
    arpu_segment: float
    frequence: float
    data_volume: float
    on_net: float
    orange: float
    tigo: float
    regularity: int


@app.post("/predict")
async def predict_sepsis(item: features):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([item.dict()])

        input_data = encoder.transform(input_data)

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Determine churn likelihood message
        churn_likelihood = "Customer is more likely to churn." if predictions[0] == 1 else "Customer is less likely to churn."

        return {"prediction": f'Churn is {predictions[0]}. {churn_likelihood}'}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
