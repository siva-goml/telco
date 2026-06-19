import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

try:
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    logger.info(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    
    logger.info("Model and preprocessor loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model artifacts: {e}")
    model = None
    preprocessor = None

class CustomerPredictionInput(BaseModel):
    Contract: str = Field(..., example="Month-to-month", description="Type of contract")
    tenure: int = Field(..., example=12, description="Number of months the customer has stayed with the company")
    TotalCharges: float = Field(..., example=1000.0, description="Total amount charged to the customer")
    MonthlyCharges: float = Field(..., example=70.0, description="Monthly charges amount")
    PaymentMethod: str = Field(..., example="Electronic check", description="Payment method used")
    OnlineSecurity: str = Field(..., example="No", description="Whether customer has Online Security option")
    TechSupport: str = Field(..., example="No", description="Whether customer has Tech Support option")
    InternetService: str = Field(..., example="Fiber optic", description="Type of internet service")
    OnlineBackup: str = Field(..., example="No", description="Whether customer has Online Backup option")
    PaperlessBilling: str = Field(..., example="Yes", description="Whether customer has Paperless Billing option")

@app.get("/")
def home():
    return {
        "message": "Welcome to the Telco Customer Churn Prediction API!",
        "status": "healthy",
    }

@app.post("/predict")
def predict_churn(data: CustomerPredictionInput):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")
        
    try:
        input_dict = data.dict()
        logger.info(f"Received prediction request for data: {input_dict}")
        
        df = pd.DataFrame([input_dict])
        
        processed_data = preprocessor.transform(df)
            
        churn_prob = float(model.predict_proba(processed_data)[0, 1])
        
        prediction = "Churn" if churn_prob >= 0.50 else "No Churn"
        
        response = {
            "prediction": prediction,
            "churn_probability": churn_prob,
            "threshold": 0.50
        }
        logger.info(f"Prediction result: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error occurred during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
