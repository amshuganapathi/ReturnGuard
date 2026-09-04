
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

from predict import predict_return_risk


app = FastAPI(
    title="ReturnGuard API",
    description="AI-powered return risk prediction API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# MODEL ARTIFACTS
# ---------------------------------------------------------

MODEL_PATH = Path("models/returnguard_final_temporal_no_item_id.cbm")
CALIBRATOR_PATH = Path("models/returnguard_calibrator_no_item_id.pkl")


# ---------------------------------------------------------
# TRANSACTION INPUT
# ---------------------------------------------------------

class Transaction(BaseModel):
    order_item_id: int = Field(
        ...,
        description="Unique order item ID"
    )

    order_date: str = Field(
        ...,
        description="Transaction date"
    )

    user_id: int = Field(
        ...,
        description="Customer ID"
    )

    item_id: int = Field(
        ...,
        description="Product ID"
    )

    item_size: str = Field(
        ...,
        description="Product size"
    )

    item_color: str = Field(
        ...,
        description="Product color"
    )

    brand_id: int = Field(
        ...,
        description="Brand ID"
    )

    item_price: float = Field(
        ...,
        gt=0,
        description="Item price"
    )

    user_title: str = Field(
        ...,
        description="Customer title"
    )

    user_dob: Optional[str] = Field(
        None,
        description="Customer date of birth"
    )

    user_state: str = Field(
        ...,
        description="Customer state"
    )

    user_reg_date: str = Field(
        ...,
        description="Customer registration date"
    )


# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "ReturnGuard",
        "status": "online",
        "model": "returnguard_final_temporal_no_item_id",
        "model_iterations": 424,
        "features": 25
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    model_loaded = MODEL_PATH.exists()
    calibrator_loaded = CALIBRATOR_PATH.exists()

    if model_loaded and calibrator_loaded:
        status = "healthy"
    else:
        status = "degraded"

    return {
        "status": status,
        "model_loaded": model_loaded,
        "calibrator_loaded": calibrator_loaded
    }


# ---------------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    try:

        transaction_data = transaction.model_dump()

        result = predict_return_risk(transaction_data)

        return {
            "success": True,
            "prediction": result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid transaction",
                "message": str(error)
            }
        )

    except Exception:

        raise HTTPException(
            status_code=503,
            detail={
                "error": "Prediction service unavailable",
                "message": "ReturnGuard could not process the transaction."
            }
        )