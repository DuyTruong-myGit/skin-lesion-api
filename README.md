# 🏥 Skin Lesion Classifier API

AI-powered skin lesion classification API with 87.34% accuracy.

## 🎯 Model Performance
- **Overall Accuracy:** 87.34%
- **Top-5 Accuracy:** 100%
- **Inference Speed:** ~2ms per image
- **Model:** YOLOv11m-cls (21MB)

## 📋 Supported Classes
- Actinic Keratoses (akiec)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic Nevus (nv)
- Vascular Lesion (vasc)

## 🚀 API Endpoints

### POST /predict
Upload image for classification

### GET /health
Health check

### GET /stats
Model statistics

### GET /classes
Supported classes info

## 🔗 Live API
https://your-app.onrender.com