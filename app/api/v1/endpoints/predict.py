from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request import URLRequest, PredictionResponse
from app.utils.features import extract_advanced_features
import numpy as np
import tensorflow as tf
import joblib

router = APIRouter()

# Global model & scaler
model = None
scaler = None

def load_model_scaler():
    global model, scaler
    if model is None:
        model = tf.keras.models.load_model("app/models/model2.h5")
        scaler = joblib.load("app/models/scaler.pkl")
    return model, scaler

@router.post("/predict", response_model=PredictionResponse)
async def predict_phishing(request: URLRequest):
    model, scaler = load_model_scaler()
    
    try:
        features_dict = extract_advanced_features(request.url)
        feature_names = ['url_length', 'special_chars', 'suspicious_keywords', 'has_https', 'num_dots',
                         'num_digits', 'domain_length', 'uses_ip', 'has_at_symbol', 'has_redirect',
                         'num_subdomains', 'suspicious_tld', 'has_http_in_domain', 'shortening_service',
                         'domain_in_path', 'num_parameters', 'prefix_suffix']
        
        # Note: Your original code had 'page_rank' — we're skipping it for live inference
        features = np.array([features_dict[f] for f in feature_names]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, features_scaled.shape[1], 1)
        
        prediction = model.predict(features_scaled, verbose=0)
        probability = float(prediction[0][0])
        is_phishing = probability > 0.5
        
        return PredictionResponse(
            url=request.url,
            phishing_probability=round(probability, 4),
            is_phishing=is_phishing,
            confidence=round(abs(probability - 0.5) * 200, 2),
            model="ABS-CNN with Self-Attention",
            features_extracted=len(features_dict)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")