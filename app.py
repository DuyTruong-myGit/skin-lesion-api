from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from PIL import Image
import io
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Skin Lesion Classifier API",
    description="AI-powered skin lesion classification with 87.34% accuracy",
    version="1.0.0"
)

# CORS - Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")
logger.info(f"Loading model from: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Class information
CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses",
        "description": "Pre-cancerous lesions caused by sun damage",
        "risk_level": "moderate"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer",
        "risk_level": "high"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growth",
        "risk_level": "low"
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Benign fibrous nodule",
        "risk_level": "low"
    },
    "mel": {
        "name": "Melanoma",
        "description": "Most serious type of skin cancer",
        "risk_level": "critical"
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "description": "Common mole, usually benign",
        "risk_level": "low"
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "Abnormal blood vessel growth",
        "risk_level": "low"
    }
}

# Response models
class PredictionItem(BaseModel):
    class_name: str
    confidence: float
    full_name: str
    risk_level: str

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    risk_level: str
    full_name: str
    description: str
    top5: List[PredictionItem]
    inference_time_ms: float

# Routes
@app.get("/")
def root():
    return {
        "api": "Skin Lesion Classifier API",
        "version": "1.0.0",
        "status": "active",
        "model": "YOLOv11m-cls",
        "accuracy": "87.34%",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "accuracy": "87.34%"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict skin lesion from image"""
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Predict
        results = model.predict(image, verbose=False)
        result = results[0]
        probs = result.probs
        
        # Main prediction
        main_class = result.names[probs.top1]
        main_conf = float(probs.top1conf)
        main_info = CLASS_INFO[main_class]
        
        # Top 5
        top5 = []
        for idx, conf in zip(probs.top5, probs.top5conf):
            cls = result.names[idx]
            top5.append({
                "class_name": cls,
                "confidence": float(conf),
                "full_name": CLASS_INFO[cls]["name"],
                "risk_level": CLASS_INFO[cls]["risk_level"]
            })
        
        # Calculate time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "prediction": main_class,
            "confidence": main_conf,
            "risk_level": main_info["risk_level"],
            "full_name": main_info["name"],
            "description": main_info["description"],
            "top5": top5,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/stats")
def get_stats():
    return {
        "model": "YOLOv11m-cls",
        "overall_accuracy": "87.34%",
        "top5_accuracy": "100%",
        "per_class": {
            "nv": "92.84%",
            "vasc": "91.67%",
            "bcc": "87.94%",
            "mel": "86.83%",
            "bkl": "85.93%",
            "df": "85.29%",
            "akiec": "66.00%"
        }
    }

@app.get("/classes")
def get_classes():
    return CLASS_INFO