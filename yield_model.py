"""
yield_model.py - Wheat and Rice Yield Forecasting for KisanAI Advisor
Loads historical Pakistani agricultural dataset and runs a Random Forest Regressor
with multi-model fallback to prevent 503 errors.
"""

from typing import Dict, Any, Optional
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

DEFAULT_DATA_PATH = "data/yield_data.csv"
DEFAULT_MODEL_PATH = "models/yield_model.joblib"


def load_yield_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads historical Pakistani agricultural dataset containing:
    crop, fertilizer_npk_kg_ha, irrigation_rounds, avg_temp_c, rainfall_mm, historical_yield_kg_ha, yield_kg_ha
    """
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception:
            pass

    # Built-in baseline records if CSV file is not present
    records = [
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 160, "irrigation_rounds": 5, "avg_temp_c": 21.0, "rainfall_mm": 85, "historical_yield_kg_ha": 3400, "yield_kg_ha": 3450},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 200, "irrigation_rounds": 6, "avg_temp_c": 20.5, "rainfall_mm": 110, "historical_yield_kg_ha": 3800, "yield_kg_ha": 3850},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 120, "irrigation_rounds": 4, "avg_temp_c": 23.0, "rainfall_mm": 60, "historical_yield_kg_ha": 2900, "yield_kg_ha": 2950},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 180, "irrigation_rounds": 6, "avg_temp_c": 21.5, "rainfall_mm": 95, "historical_yield_kg_ha": 3600, "yield_kg_ha": 3620},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 140, "irrigation_rounds": 5, "avg_temp_c": 22.0, "rainfall_mm": 75, "historical_yield_kg_ha": 3200, "yield_kg_ha": 3250},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 220, "irrigation_rounds": 7, "avg_temp_c": 19.8, "rainfall_mm": 130, "historical_yield_kg_ha": 4100, "yield_kg_ha": 4180},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 100, "irrigation_rounds": 3, "avg_temp_c": 24.5, "rainfall_mm": 45, "historical_yield_kg_ha": 2600, "yield_kg_ha": 2550},
        {"crop": "Wheat", "fertilizer_npk_kg_ha": 170, "irrigation_rounds": 5, "avg_temp_c": 21.2, "rainfall_mm": 90, "historical_yield_kg_ha": 3500, "yield_kg_ha": 3550},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 150, "irrigation_rounds": 14, "avg_temp_c": 31.0, "rainfall_mm": 420, "historical_yield_kg_ha": 3200, "yield_kg_ha": 3300},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 180, "irrigation_rounds": 16, "avg_temp_c": 30.5, "rainfall_mm": 480, "historical_yield_kg_ha": 3600, "yield_kg_ha": 3650},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 110, "irrigation_rounds": 12, "avg_temp_c": 32.0, "rainfall_mm": 350, "historical_yield_kg_ha": 2800, "yield_kg_ha": 2750},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 190, "irrigation_rounds": 18, "avg_temp_c": 29.8, "rainfall_mm": 510, "historical_yield_kg_ha": 3850, "yield_kg_ha": 3900},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 130, "irrigation_rounds": 13, "avg_temp_c": 31.5, "rainfall_mm": 390, "historical_yield_kg_ha": 3000, "yield_kg_ha": 3050},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 200, "irrigation_rounds": 17, "avg_temp_c": 30.0, "rainfall_mm": 530, "historical_yield_kg_ha": 4000, "yield_kg_ha": 4080},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 100, "irrigation_rounds": 11, "avg_temp_c": 33.0, "rainfall_mm": 310, "historical_yield_kg_ha": 2500, "yield_kg_ha": 2480},
        {"crop": "Rice", "fertilizer_npk_kg_ha": 160, "irrigation_rounds": 15, "avg_temp_c": 30.8, "rainfall_mm": 440, "historical_yield_kg_ha": 3400, "yield_kg_ha": 3420},
    ]
    return pd.DataFrame(records)


def train_and_save_model(data_path: str = DEFAULT_DATA_PATH, model_path: str = DEFAULT_MODEL_PATH):
    """
    Trains a Random Forest Regressor and serializes it to models/yield_model.joblib
    """
    df = load_yield_data(data_path)
    df["is_wheat"] = (df["crop"] == "Wheat").astype(int)

    features = ["is_wheat", "fertilizer_npk_kg_ha", "irrigation_rounds", "avg_temp_c", "rainfall_mm", "historical_yield_kg_ha"]
    X = df[features]
    y = df["yield_kg_ha"]

    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        joblib.dump(model, model_path)
    except Exception:
        pass
    return model


def load_yield_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads saved Random Forest model, or trains one if file does not exist.
    """
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    return train_and_save_model(DEFAULT_DATA_PATH, model_path)


def predict_crop_yield(
    crop: str,
    area_value: float,
    area_unit: str,
    fertilizer_npk: float,
    irrigation_rounds: int,
    avg_temp: float,
    rainfall: float,
    historical_yield: float,
    model=None
) -> Dict[str, Any]:
    """
    Predicts crop harvest in Tonnes, Maunds, and kg/ha.
    1 maund (من) = 40 kg
    """
    if area_unit.lower() == "acres":
        area_ha = area_value / 2.47105
        area_acres = area_value
    else:
        area_ha = area_value
        area_acres = area_value * 2.47105

    if model is None:
        model = load_yield_model()

    is_wheat = 1 if crop == "Wheat" else 0
    features = np.array([[is_wheat, fertilizer_npk, irrigation_rounds, avg_temp, rainfall, historical_yield]])
    
    try:
        predicted_kg_ha = float(model.predict(features)[0])
    except Exception:
        predicted_kg_ha = 3450.0 if crop == "Wheat" else 3300.0

    # Safeguard bounds based on Pakistani agro-climatic averages
    predicted_kg_ha = max(1600.0, min(6500.0, predicted_kg_ha))

    total_kg = predicted_kg_ha * area_ha
    total_tonnes = total_kg / 1000.0
    total_maunds = total_kg / 40.0

    return {
        "crop": crop,
        "area_hectares": round(area_ha, 2),
        "area_acres": round(area_acres, 1),
        "yield_per_hectare_kg": int(round(predicted_kg_ha)),
        "total_yield_kg": int(round(total_kg)),
        "total_yield_tonnes": round(total_tonnes, 2),
        "total_yield_maunds": round(total_maunds, 1),
        "warning": "NOTICE: This forecast is an estimate generated by a scikit-learn Random Forest model. Actual harvest depends on certified seed quality, timely weed suppression, and localized weather shocks."
    }


def get_yield_gemini_explanation(crop: str, area_ha: float, total_tonnes: float, yield_per_ha: int) -> Dict[str, Any]:
    """
    Generates yield assessment with multi-model fallback to survive 503 high demand spikes.
    Falls back gracefully to benchmark agronomic knowledgebase if all models are busy.
    """
    standard_explanation = (
        f"A forecasted harvest of {total_tonnes} metric tonnes ({yield_per_ha} kg/ha) reflects "
        f"productive management for {crop} under Pakistani growing conditions. To protect this yield, "
        f"ensure timely weed suppression during early vegetative growth and monitor for pest flare-ups."
    )
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {
            "explanation": standard_explanation,
            "source": "Agronomic Benchmark Rules (Predefined)"
        }
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = (
            f"You are an agricultural advisor for smallholder farmers in Pakistan. "
            f"Crop: {crop}. Cultivated Area: {area_ha} hectares. "
            f"Forecasted Total Harvest: {total_tonnes} metric tonnes ({yield_per_ha} kg/ha). "
            f"Provide a concise 2-sentence encouraging assessment of this yield level compared to typical Pakistani productivity, "
            f"and one actionable tip for optimizing harvest (e.g. weed control or balanced fertilizer). "
            f"Keep the language very simple, respectful, and direct."
        )
        
        candidate_models = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"]
        text_output = None
        for mod in candidate_models:
            try:
                response = client.models.generate_content(
                    model=mod,
                    contents=prompt,
                )
                if response.text and response.text.strip():
                    text_output = response.text.strip()
                    break
            except Exception:
                continue
        
        if not text_output:
            text_output = standard_explanation
            
        return {
            "explanation": text_output,
            "source": "Gemini AI Advisor" if text_output != standard_explanation else "Agronomic Benchmark Rules (Predefined)"
        }
    except Exception:
        return {
            "explanation": standard_explanation,
            "source": "Agronomic Benchmark Rules (Predefined)"
        }
