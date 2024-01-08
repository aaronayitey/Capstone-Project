from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load your pre-trained model and any necessary preprocessing pipeline
# Make sure to update the file paths accordingly
pipeline = joblib.load('./churn_pipeline.joblib')
model = joblib.load('./random_forest_model.joblib')
encoder = joblib.load('./label_encoder.joblib')

app = FastAPI()

# Define the Pydantic model for the input features
class Features(BaseModel):
    Duration: str
    Top_up_Amount: float
    Refill_Frequency: float
    Monthly_Income: float
    Income_over_90_Days: float
    Income_Frequency: float
    Number_of_Connections: float
    Inter_Expresso_Call: float
    Call_to_Orange: float
    Call_to_Tigo: float
    Top_Pack_Package_Activation_frequency: float

# Mapping between API names and actual model names
api_to_model_mapping = {
    "Duration": "tenure",
    "Top_up_Amount": "montant",
    "Refill_Frequency": "frequence_rech",
    "Monthly_Income": "revenue",
    "Income_over_90_Days": "arpu_segment",
    "Income_Frequency": "frequence",
    "Number_of_Connections": "data_volume",
    "Inter_Expresso_Call": "on_net",
    "Call_to_Orange": "orange",
    "Call_to_Tigo": "tigo",
    "Top_Pack_Package_Activation_frequency": "regularity",
}

# Endpoint to make predictions
@app.post("/predict")
async def predict_churn(input_data: Features):
    
    try:
       # Access the 'Duration' variable directly from the input_data parameter
        duration_value = input_data.Duration

        # Check if the duration_value is in the label encoder's classes
        if duration_value not in encoder.classes_:
            raise HTTPException(status_code=422, detail=f"Invalid 'Duration' value: {duration_value}")
        else:
            # Transform 'Duration' using the label encoder
            input_data_encoded = input_data.copy()
            input_data_encoded.Duration = encoder.transform([duration_value])[0]

        # Convert input data to DataFrame
        input_data = pd.DataFrame([input_data.dict()])

        # Map API names to actual model names
        input_data.columns = [api_to_model_mapping.get(col, col) for col in input_data.columns]

        # Make predictions using the model
        predictions = pipeline.predict(input_data)

        # Customize the response based on your model output
        # For example, you can return a probability score or a specific class
        return {"prediction": float(predictions[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

