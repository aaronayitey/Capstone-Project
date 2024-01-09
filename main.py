# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# # Load your pre-trained model and any necessary preprocessing pipeline
# # Make sure to update the file paths accordingly
# encoder = joblib.load('./transform_encode.joblib')
# model = joblib.load('./final_xgb_model.joblib')
# # encoder = joblib.load('./label_encoder.joblib')

# app = FastAPI()

# # Define the Pydantic model for the input features
# class Features(BaseModel):
#     Duration: str
#     Top_up_Amount: float
#     Refill_Frequency: float
#     Monthly_Income: float
#     Income_over_90_Days: float
#     Income_Frequency: float
#     Number_of_Connections: float
#     Inter_Expresso_Call: float
#     Call_to_Orange: float
#     Call_to_Tigo: float
#     Top_Pack_Package_Activation_frequency: float

# # Mapping between API names and actual model names
# api_to_model_mapping = {
#     "Duration": "tenure",
#     "Top_up_Amount": "montant",
#     "Refill_Frequency": "frequence_rech",
#     "Monthly_Income": "revenue",
#     "Income_over_90_Days": "arpu_segment",
#     "Income_Frequency": "frequence",
#     "Number_of_Connections": "data_volume",
#     "Inter_Expresso_Call": "on_net",
#     "Call_to_Orange": "orange",
#     "Call_to_Tigo": "tigo",
#     "Top_Pack_Package_Activation_frequency": "regularity",
# }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the pre-trained model
model = joblib.load("./final_xgb_model.joblib") 
encoder = joblib.load("./transform_encode.joblib")
# Define the FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class ChurnPredictionInput(BaseModel):
    montant: float
    frequence_rech: int
    revenue: float
    arpu_segment: float
    frequence: float
    data_volume: float
    on_net: float
    orange: float
    tigo: float
    regularity: float
    tenure: str 

# Endpoint for making predictions
@app.post("/predict")
def predict_churn(input_data: ChurnPredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Identify numeric and categorical columns
        numeric_cols = input_df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = input_df.select_dtypes(include=['object']).columns


        X_input_transformed = encoder.transform(input_df)

        # Make predictions
        prediction = model.predict(X_input_transformed)

        # Return the prediction
        return {"prediction": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to make predictions
# @app.post("/predict")
# async def predict_sepsis(item: Features):
#     try:
#         # Convert input data to DataFrame
#         input_data = pd.DataFrame([item.dict()])
#         # Map API names to actual model names
#         # input_data.columns = [api_to_model_mapping.get(col, col) for col in input_data.columns]

#         input_data = encoder.transform(input_data)

        

#         # Make predictions using the model
#         predictions = model.predict(input_data)

#         # Customize the response based on your model output
#         # For example, you can return a probability score or a specific class
#         return {"prediction": float(predictions[0])}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

