
# KisanAI Advisor 🌾 (پاکستان کسان مشاورتی سروس)

Mobile-friendly agricultural decision support application for smallholder farmers and agricultural extension officers across Pakistan (Punjab, Sindh, Khyber Pakhtunkhwa, Balochistan).

## Features
1. **Crop Disease Detection**: MobileNetV2 leaf disease identification for Wheat and Rice with Urdu/English agricultural extension advice.
2. **Weather-Linked Irrigation Alerts**: Real-time Open-Meteo forecasts and FAO ET₀ evaporation rules.
3. **Wheat & Rice Yield Forecasting**: scikit-learn Random Forest model outputting in Tonnes, kg, and Pakistani Maunds (من).

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
