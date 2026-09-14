"""
app.py - KisanAI Advisor 🌾
Streamlit Web Application for Pakistani Smallholder Farmers & Extension Officers
"""

import os
import streamlit as st
from PIL import Image

# Import domain modules (strict one-way dependency)
from disease import predict_disease, load_disease_model
from irrigation import PAKISTANI_DISTRICTS, load_weather_data, evaluate_irrigation_need
from yield_model import predict_crop_yield, load_yield_model

# Streamlit Page Setup
st.set_page_config(
    page_title="KisanAI Advisor 🌾 Pakistan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean mobile-friendly layout
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #065f46; margin-bottom: 0px; }
    .sub-title { font-size: 0.95rem; color: #047857; margin-bottom: 1.5rem; }
    .card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
    .metric-val { font-size: 1.8rem; font-weight: 800; color: #0f172a; }
</style>
""", unsafe_allow_html=True)


def get_gemini_explanation(prompt: str) -> str:
    """
    Optional Gemini explanation helper using official google-genai SDK.
    Reads GEMINI_API_KEY from st.secrets or environment.
    Falls back gracefully if key is absent.
    """
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return ""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        return ""


# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🌾 KisanAI Advisor")
    st.markdown("**پاکستان کسان مشاورتی سروس**")
    st.caption("Agricultural Decision Support for Pakistani Wheat & Rice Growers")
    selected_tab = st.radio(
        "Choose Advisory Tool:",
        ["🌿 Crop Disease Detection", "💧 Weather & Irrigation Alerts", "📈 Yield Forecasting"],
        index=0
    )
    st.divider()
    st.markdown("### 🇵🇰 Coverage Areas")
    st.caption("Punjab, Sindh, Khyber Pakhtunkhwa, Balochistan")
    st.info("💡 **Farmer Tip:** Always verify field symptoms before applying chemical sprays.")

# -------------------------------------------------------------
# TAB 1: CROP DISEASE DETECTION
# -------------------------------------------------------------
if selected_tab == "🌿 Crop Disease Detection":
    st.markdown("<h1 class='main-title'>🌿 Crop Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>فصل کی بیماری کی فوری تشخیص — Screen leaf photos for fungal and bacterial infections in Wheat and Rice</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 1. Upload or Snap Leaf Photo")
        crop_choice = st.selectbox("Select Crop Type / فصل منتخب کریں:", ["Wheat (گندم)", "Rice (چاول)", "Other Crop"], index=0)
        crop_clean = "Wheat" if "Wheat" in crop_choice else ("Rice" if "Rice" in crop_choice else "Other")

        uploaded_file = st.file_uploader("Upload leaf photo (JPG/PNG):", type=["jpg", "jpeg", "png"])
        camera_file = st.camera_input("Or take a photo using device camera:")

        active_file = uploaded_file or camera_file

        if active_file is not None:
            image = Image.open(active_file)
            st.image(image, caption="Selected Leaf Image", use_container_width=True)
            diagnose_btn = st.button("Diagnose Leaf Disease / بیماری کی تشخیص کریں", type="primary", use_container_width=True)
        else:
            diagnose_btn = False
            st.info("👆 Upload or snap a leaf photo to begin diagnosis.")

    with col2:
        st.markdown("### 2. Diagnosis Result & Extension Advisory")
        if diagnose_btn and active_file is not None:
            with st.spinner("Analyzing leaf with MobileNetV2..."):
                disease_model = load_disease_model()
                res = predict_disease(image, model=disease_model, crop_hint=crop_clean)

                if res["is_uncertain"]:
                    st.warning(f"⚠️ **Low Confidence Result ({res['confidence']*100:.1f}%)**: Photo may be blurry or poorly lit. Retake a sharp close-up.")
                else:
                    st.success(f"✅ **Detected Condition:** {res['disease_name']}")

                st.metric("Model Confidence", f"{res['confidence']*100:.1f}%")
                if res["scientific_name"] != "N/A":
                    st.caption(f"*Scientific Name:* {res['scientific_name']}")

                st.markdown(f"**Description:** {res['summary']}")

                # Gemini explanation if available
                gem_prompt = f"You are an agricultural advisor for smallholder Pakistani farmers. A farmer's {crop_clean} crop leaf shows '{res['disease_name']}'. Explain in 2 plain sentences what this means and 2 practical immediate steps."
                ai_exp = get_gemini_explanation(gem_prompt)
                if ai_exp:
                    st.markdown(f"**🤖 Advisor Notes:** {ai_exp}")

                st.markdown("#### Immediate Recommended Steps / کسان کے لیے فوری اقدامات:")
                for step in res["steps"]:
                    st.markdown(f"- {step}")

                st.error(res["warning"])
        else:
            st.markdown("<div class='card'>Inference results and agronomic guidance will appear here after evaluation.</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: WEATHER & IRRIGATION ALERTS
# -------------------------------------------------------------
elif selected_tab == "💧 Weather & Irrigation Alerts":
    st.markdown("<h1 class='main-title'>💧 Weather-Linked Irrigation Alerts</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>موسمی حالات کے مطابق آبپاشی کی رہنمائی — Live Open-Meteo forecasts and FAO ET₀ evaporation rules</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 1. Farm & Location Parameters")
        irrig_crop = st.radio("Crop Type / فصل:", ["Wheat (گندم - Rabi)", "Rice (چاول - Kharif)"], horizontal=True)
        crop_clean = "Wheat" if "Wheat" in irrig_crop else "Rice"

        loc_type = st.radio("Location Mode:", ["Major District", "Custom GPS Coordinates"], horizontal=True)

        if loc_type == "Major District":
            district_names = [f"{d['name']} ({d['province']})" for d in PAKISTANI_DISTRICTS]
            selected_dist_name = st.selectbox("Select Agricultural District:", district_names, index=0)
            selected_dist = next(d for d in PAKISTANI_DISTRICTS if f"{d['name']} ({d['province']})" == selected_dist_name)
            lat, lon = selected_dist["lat"], selected_dist["lon"]
            st.caption(f"Coordinates: {lat:.4f}°N, {lon:.4f}°E ({selected_dist['zone']})")
        else:
            c_lat, c_lon = st.columns(2)
            with c_lat:
                lat = st.number_input("Latitude (°N):", value=31.4187, format="%.4f")
            with c_lon:
                lon = st.number_input("Longitude (°E):", value=73.0791, format="%.4f")

        use_soil = st.checkbox("Include Soil Moisture Sensor Reading / مٹی کی نمی")
        soil_val = None
        if use_soil:
            soil_val = st.slider("Soil Moisture Level (%):", min_value=10, max_value=90, value=42)

        check_irrig = st.button("Check Irrigation Need / آبپاشی کی جانچ کریں", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 2. Live Weather Data & Recommendation")
        if check_irrig:
            with st.spinner("Querying Open-Meteo meteorological feed..."):
                weather = load_weather_data(lat, lon)
                if weather:
                    w1, w2, w3 = st.columns(3)
                    w1.metric("Max Temp", f"{weather['max_temp']}°C")
                    w2.metric("3-Day Rain", f"{weather['forecast_3day_rain_mm']} mm")
                    w3.metric("ET₀ Water Loss", f"{weather['et0_mm']} mm/d")

                    alert = evaluate_irrigation_need(crop_clean, weather, soil_val)

                    if alert["recommendation"] == "Rainfall may reduce irrigation need":
                        st.success(f"✅ **Advisory:** {alert['recommendation']}")
                    else:
                        st.warning(f"💧 **Advisory:** {alert['recommendation']}")

                    st.markdown(f"**Urgency:** `{alert['urgency']}`")
                    st.markdown(f"**Agronomic Reason:** {alert['reason']}")
                    st.markdown(f"**Guidance:** {alert['guidance']}")

                    # Gemini explanation
                    gem_prompt = f"You are an irrigation expert in Pakistan. Crop: {crop_clean}. Recommendation: {alert['recommendation']}. Weather: Max {weather['max_temp']}C, 3-day rain {weather['forecast_3day_rain_mm']}mm. Give 2 sentences of simple advice to the farmer."
                    ai_exp = get_gemini_explanation(gem_prompt)
                    if ai_exp:
                        st.markdown(f"**🤖 Advisor Notes:** {ai_exp}")

                    st.caption("⚠️ **DISCLAIMER:** Advisory recommendations calculated from forecast models. Always physically inspect soil firmness before starting tube-wells.")
                else:
                    st.error("Failed to retrieve weather data from Open-Meteo. Please try again.")
        else:
            st.markdown("<div class='card'>Click 'Check Irrigation Need' to fetch real-time atmospheric data.</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: YIELD FORECASTING
# -------------------------------------------------------------
elif selected_tab == "📈 Yield Forecasting":
    st.markdown("<h1 class='main-title'>📈 Wheat & Rice Yield Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>گندم اور چاول کی متوقع پیداوار کا تخمینہ — Random Forest estimates in Tonnes, kg, and Pakistani Maunds (من)</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 1. Farm Inputs & Management")
        yield_crop = st.radio("Target Crop / فصل:", ["Wheat (گندم)", "Rice (چاول)"], horizontal=True)
        crop_clean = "Wheat" if "Wheat" in yield_crop else "Rice"

        u1, u2 = st.columns(2)
        with u1:
            unit = st.selectbox("Area Unit:", ["Acres (ایکڑ)", "Hectares (ہیکٹر)"])
        with u2:
            area_val = st.number_input("Cultivated Area:", min_value=0.5, max_value=500.0, value=5.0, step=0.5)

        f1, f2 = st.columns(2)
        with f1:
            fertilizer = st.number_input("NPK Fertilizer (kg/ha):", min_value=50.0, max_value=350.0, value=160.0 if crop_clean == "Wheat" else 150.0, step=10.0)
        with f2:
            irrig_rounds = st.number_input("Planned Irrigation Rounds:", min_value=1, max_value=25, value=5 if crop_clean == "Wheat" else 14, step=1)

        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.number_input("Avg Temp (°C):", value=21.0 if crop_clean == "Wheat" else 30.5, step=0.5)
        with c2:
            rain = st.number_input("Season Rain (mm):", value=85.0 if crop_clean == "Wheat" else 420.0, step=5.0)
        with c3:
            hist_yield = st.number_input("Hist. Yield (kg/ha):", value=3400.0 if crop_clean == "Wheat" else 3200.0, step=50.0)

        predict_btn = st.button("Estimate Harvest / متوقع پیداوار جانچیں", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 2. Projected Harvest & Breakdown")
        if predict_btn:
            with st.spinner("Running Random Forest Regressor..."):
                pred = predict_crop_yield(
                    crop=crop_clean,
                    area_value=area_val,
                    area_unit="acres" if "Acres" in unit else "hectares",
                    fertilizer_npk=fertilizer,
                    irrigation_rounds=irrig_rounds,
                    avg_temp=temp,
                    rainfall=rain,
                    historical_yield=hist_yield
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Tonnes", f"{pred['total_yield_tonnes']} t")
                m2.metric("Total Maunds", f"{pred['total_yield_maunds']:,} من")
                m3.metric("Productivity", f"{pred['yield_per_hectare_kg']} kg/ha")

                st.markdown(f"""
                <div class='card'>
                    <h4>🌾 Forecast Summary for {pred['crop']}:</h4>
                    <p>Estimated total production of <strong>{pred['total_yield_tonnes']} Metric Tonnes</strong> (~{pred['total_yield_kg']:,} kg) across {pred['area_hectares']} hectares (~{pred['area_acres']} acres).</p>
                    <p>Equivalent to approximately <strong>{pred['total_yield_maunds']:,} Pakistani Maunds (من)</strong> (standard 40 kg per maund).</p>
                </div>
                """, unsafe_allow_html=True)

                # Gemini explanation
                gem_prompt = f"You are an agricultural advisor in Pakistan. A farmer is projected to produce {pred['total_yield_tonnes']} tonnes of {crop_clean} ({pred['yield_per_hectare_kg']} kg/ha). Give a 2-sentence encouraging note on productivity and one tip for harvest timing."
                ai_exp = get_gemini_explanation(gem_prompt)
                if ai_exp:
                    st.markdown(f"**🤖 Advisor Notes:** {ai_exp}")

                st.caption(pred["warning"])
        else:
            st.markdown("<div class='card'>Enter field parameters and click 'Estimate Harvest' to generate predictions.</div>", unsafe_allow_html=True)
