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

# CORS
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
        "risk_level": "moderate",
        "recommendation": "Consult dermatologist for evaluation and possible treatment"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer",
        "risk_level": "high",
        "recommendation": "Schedule dermatologist appointment soon for biopsy"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growth",
        "risk_level": "low",
        "recommendation": "Generally harmless, monitor for changes"
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Benign fibrous nodule",
        "risk_level": "low",
        "recommendation": "Usually no treatment needed, can be removed if bothersome"
    },
    "mel": {
        "name": "Melanoma",
        "description": "Most serious type of skin cancer",
        "risk_level": "critical",
        "recommendation": "URGENT: See dermatologist/oncologist immediately"
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "description": "Common mole, usually benign",
        "risk_level": "low",
        "recommendation": "Monitor for changes using ABCDE rule"
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "Abnormal blood vessel growth",
        "risk_level": "low",
        "recommendation": "Consult dermatologist if concerned about appearance"
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
    recommendation: str
    top5: List[PredictionItem]
    inference_time_ms: float
    timestamp: str

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "api": "Skin Lesion Classifier API",
        "version": "1.0.0",
        "status": "active",
        "model": "YOLOv11m-cls",
        "performance": {
            "accuracy": "87.34%",
            "top5_accuracy": "100%",
            "inference_speed": "~2ms"
        },
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "stats": "GET /stats",
            "classes": "GET /classes",
            "docs": "GET /docs"
        },
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "accuracy": "87.34%"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict skin lesion from uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        Prediction results with confidence scores and medical recommendations
    """
    
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Predict
        results = model.predict(image, verbose=False)
        result = results[0]
        probs = result.probs
        class_names = result.names
        
        # Main prediction
        main_class = class_names[probs.top1]
        main_conf = float(probs.top1conf)
        main_info = CLASS_INFO[main_class]
        
        # Top 5 predictions
        top5 = []
        for idx, conf in zip(probs.top5, probs.top5conf):
            cls = class_names[idx]
            info = CLASS_INFO[cls]
            top5.append({
                "class_name": cls,
                "confidence": float(conf),
                "full_name": info["name"],
                "risk_level": info["risk_level"]
            })
        
        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Prediction: {main_class} ({main_conf:.2%}), Time: {inference_time:.2f}ms")
        
        return {
            "success": True,
            "prediction": main_class,
            "confidence": main_conf,
            "risk_level": main_info["risk_level"],
            "full_name": main_info["name"],
            "description": main_info["description"],
            "recommendation": main_info["recommendation"],
            "top5": top5,
            "inference_time_ms": round(inference_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/stats")
def get_stats():
    """Get model statistics and performance metrics"""
    return {
        "model": "YOLOv11m-cls",
        "parameters": "10.4M",
        "model_size": "20.9MB",
        "overall_performance": {
            "test_accuracy": "87.34%",
            "top5_accuracy": "100%",
            "inference_speed": "~2ms per image"
        },
        "per_class_accuracy": {
            "nv (Melanocytic Nevus)": "92.84%",
            "vasc (Vascular Lesion)": "91.67%",
            "bcc (Basal Cell Carcinoma)": "87.94%",
            "mel (Melanoma)": "86.83%",
            "bkl (Benign Keratosis)": "85.93%",
            "df (Dermatofibroma)": "85.29%",
            "akiec (Actinic Keratoses)": "66.00%"
        },
        "training_info": {
            "dataset_size": "23,125 images",
            "training_time": "3.76 hours",
            "epochs": "100"
        }
    }

@app.get("/classes")
def get_classes():
    """Get information about all supported classes"""
    return {
        "total_classes": len(CLASS_INFO),
        "classes": CLASS_INFO
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)