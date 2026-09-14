"""
irrigation.py - Weather-Linked Irrigation Module for KisanAI Advisor
Retrieves Open-Meteo forecasts and applies FAO ET0 agronomic irrigation rules.
"""

from typing import Dict, Any, Optional
import requests

PAKISTANI_DISTRICTS = [
    {"id": "fsd", "name": "Faisalabad", "province": "Punjab", "lat": 31.4187, "lon": 73.0791, "zone": "Central Mixed Crop Zone"},
    {"id": "lhr", "name": "Sheikhupura / Lahore", "province": "Punjab", "lat": 31.7167, "lon": 73.9850, "zone": "Kalar Rice-Wheat Belt"},
    {"id": "mul", "name": "Multan", "province": "Punjab", "lat": 30.1575, "lon": 71.5249, "zone": "Cotton-Wheat Belt"},
    {"id": "guj", "name": "Gujranwala", "province": "Punjab", "lat": 32.1877, "lon": 74.1945, "zone": "Basmati Rice Tract"},
    {"id": "shw", "name": "Sahiwal", "province": "Punjab", "lat": 30.6682, "lon": 73.1114, "zone": "Intensive Irrigated Plains"},
    {"id": "bwp", "name": "Bahawalpur", "province": "Punjab", "lat": 29.3956, "lon": 71.6836, "zone": "Southern Arid Irrigated Zone"},
    {"id": "suk", "name": "Sukkur", "province": "Sindh", "lat": 27.7052, "lon": 68.8574, "zone": "Upper Sindh Indus Basin"},
    {"id": "lrk", "name": "Larkana", "province": "Sindh", "lat": 27.5589, "lon": 68.2120, "zone": "Sindh Rice Canal Tract"},
    {"id": "hyd", "name": "Hyderabad / Tando Jam", "province": "Sindh", "lat": 25.3960, "lon": 68.3578, "zone": "Lower Sindh Agronomic Area"},
    {"id": "pes", "name": "Peshawar", "province": "Khyber Pakhtunkhwa", "lat": 34.0151, "lon": 71.5249, "zone": "Peshawar Valley Irrigated Zone"},
    {"id": "mar", "name": "Mardan", "province": "Khyber Pakhtunkhwa", "lat": 34.1989, "lon": 72.0404, "zone": "Swat Canal Command Area"},
    {"id": "qta", "name": "Quetta / Pishin", "province": "Balochistan", "lat": 30.1798, "lon": 66.9750, "zone": "Upland Valley Farming"}
]


def load_weather_data(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """
    Fetches real-time weather and 3-day forecasts from Open-Meteo API.
    Does not require an API key.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude:.4f}&longitude={longitude:.4f}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
        f"&timezone=auto&forecast_days=3"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        precip_list = daily.get("precipitation_sum", [0.0, 0.0, 0.0])
        forecast_3day_rain = sum(precip_list[:3]) if precip_list else 0.0

        return {
            "current_temp": current.get("temperature_2m", 25.0),
            "humidity": current.get("relative_humidity_2m", 50.0),
            "today_rain_mm": precip_list[0] if precip_list else 0.0,
            "forecast_3day_rain_mm": round(forecast_3day_rain, 1),
            "max_temp": daily.get("temperature_2m_max", [28.0])[0],
            "min_temp": daily.get("temperature_2m_min", [16.0])[0],
            "et0_mm": daily.get("et0_fao_evapotranspiration", [3.8])[0]
        }
    except Exception as e:
        print(f"Open-Meteo request failed: {e}")
        return None


def evaluate_irrigation_need(
    crop: str,
    weather: Optional[Dict[str, Any]],
    soil_moisture: Optional[float] = None
) -> Dict[str, Any]:
    """
    Applies agronomic rules for irrigation recommendation:
    - 'Rainfall may reduce irrigation need'
    - 'Water may be needed'
    - 'Insufficient data'
    """
    if weather is None:
        return {
            "recommendation": "Insufficient data",
            "urgency": "None",
            "reason": "Could not connect to Open-Meteo meteorological feed.",
            "guidance": "Verify internet connectivity and try again."
        }

    rain_3day = weather.get("forecast_3day_rain_mm", 0.0)
    today_rain = weather.get("today_rain_mm", 0.0)
    et0 = weather.get("et0_mm", 3.8)
    max_temp = weather.get("max_temp", 26.0)

    if soil_moisture is not None and soil_moisture >= 65.0:
        return {
            "recommendation": "Rainfall may reduce irrigation need",
            "urgency": "Low / Postpone",
            "reason": f"Soil moisture is high ({soil_moisture}%). Root zone is adequately wet.",
            "guidance": f"Postpone tube-well pumping. Saturated soil can induce root rot in {crop} and wastes fuel."
        }

    if rain_3day >= 10.0 or today_rain >= 8.0:
        return {
            "recommendation": "Rainfall may reduce irrigation need",
            "urgency": "Low / Postpone",
            "reason": f"Substantial rainfall forecasted ({rain_3day} mm over next 3 days).",
            "guidance": "Delay planned irrigation. Upcoming rains will replenish the root zone and save canal allocations."
        }

    if rain_3day >= 5.0 and (soil_moisture is None or soil_moisture >= 40.0):
        return {
            "recommendation": "Rainfall may reduce irrigation need",
            "urgency": "Low / Routine",
            "reason": f"Moderate rainfall ({rain_3day} mm) will offset daily crop evapotranspiration ({et0} mm/day).",
            "guidance": "Hold off on full irrigation today. Check field firmness again tomorrow morning."
        }

    if soil_moisture is not None and soil_moisture < 35.0:
        return {
            "recommendation": "Water may be needed",
            "urgency": "High",
            "reason": f"Soil moisture is low ({soil_moisture}%) with dry forecast ({rain_3day} mm).",
            "guidance": f"Schedule irrigation promptly. {crop} is entering water stress. Irrigate in early morning or evening."
        }

    if crop == "Rice":
        return {
            "recommendation": "Water may be needed",
            "urgency": "Normal",
            "reason": f"Rice paddies require standing water; rain is minimal ({rain_3day} mm) vs ET₀ {et0} mm/day.",
            "guidance": "Maintain 2-3 inch standing water depth. Inspect field bunds (watbandi) for leaks."
        }

    if max_temp > 28.0 or et0 >= 4.0:
        return {
            "recommendation": "Water may be needed",
            "urgency": "Medium",
            "reason": f"Warm weather ({max_temp}°C) and high ET₀ ({et0} mm/day) are drying topsoil.",
            "guidance": "Irrigate if wheat is at critical stages (Crown Root Initiation at 21-25 days or flowering)."
        }

    return {
        "recommendation": "Water may be needed",
        "urgency": "Low / Routine",
        "reason": f"Mild weather ({max_temp}°C, ET₀ {et0} mm) with dry forecast ({rain_3day} mm).",
        "guidance": "Inspect crop visual vigor. Irrigate only if leaves show midday rolling."
    }
