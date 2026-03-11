import os
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "sla_svm_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# Country encoding inferred from your dataset
COUNTRY_MAP = {
    "China": 0,
    "France": 1,
    "Germany": 2,
    "India": 3,
    "Mexico": 4,
    "Poland": 5,
    "UAE": 6,
    "UK": 7,
    "USA": 8,
    "Vietnam": 9,
}

# Helper defaults based on your dataset patterns
MODE_STATS = {
    "Rail": {"mode_avg_delay": 0.2960569495632857, "mode_sla_breach_rate": 0.46958146109961124},
    "Road": {"mode_avg_delay": 0.28975923609160775, "mode_sla_breach_rate": 0.4728342737508617},
    "Sea": {"mode_avg_delay": 0.2950015126558876, "mode_sla_breach_rate": 0.4726881575851289},
    "Unknown": {"mode_avg_delay": 0.2890415124942977, "mode_sla_breach_rate": 0.4719955395610523},
}

# Simple carrier defaults for now
CARRIER_DEFAULTS = {
    "Carrier_A": {"carrier_avg_delay": 0.28, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_B": {"carrier_avg_delay": 0.29, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_C": {"carrier_avg_delay": 0.30, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_D": {"carrier_avg_delay": 0.31, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_E": {"carrier_avg_delay": 0.30, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_F": {"carrier_avg_delay": 0.29, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_G": {"carrier_avg_delay": 0.31, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.125},
    "Carrier_H": {"carrier_avg_delay": 0.26, "carrier_sla_breach_rate": 0.47, "carrier_volume_share": 0.124},
}

# Route defaults for now
DEFAULT_ROUTE_AVG_DELAY = 0.30
DEFAULT_ROUTE_SLA_BREACH_RATE = 0.47


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_region_flags(origin_country, destination_country):
    apac_countries = {"China", "India", "Vietnam"}
    emea_countries = {"France", "Germany", "Poland", "UAE", "UK"}

    involved = {origin_country, destination_country}

    region_apac = int(len(involved.intersection(apac_countries)) > 0)
    region_emea = int(len(involved.intersection(emea_countries)) > 0)

    return region_apac, region_emea


def build_feature_dict(form_data):
    # Raw user inputs
    shipment_weight_kg = to_float(form_data.get("shipment_weight_kg"))
    shipment_volume_cbm = to_float(form_data.get("shipment_volume_cbm"))
    priority_flag = to_int(form_data.get("priority_flag"))
    fragile_flag = to_int(form_data.get("fragile_flag"))
    temperature_control_flag = to_int(form_data.get("temperature_control_flag"))
    planned_delivery_days = to_float(form_data.get("planned_delivery_days"))
    shipping_cost_usd = to_float(form_data.get("shipping_cost_usd"))
    fuel_surcharge_pct = to_float(form_data.get("fuel_surcharge_pct"))
    shipment_value_usd = to_float(form_data.get("shipment_value_usd"))
    insurance_flag = to_int(form_data.get("insurance_flag"))

    origin_country = form_data.get("origin_country", "China")
    destination_country = form_data.get("destination_country", "China")
    shipping_mode = form_data.get("shipping_mode", "Unknown")
    carrier = form_data.get("carrier", "Carrier_A")

    # Encoded / mapped
    origin_country_encoded = COUNTRY_MAP.get(origin_country, 0)
    destination_country_encoded = COUNTRY_MAP.get(destination_country, 0)

    shipping_mode_rail = int(shipping_mode == "Rail")
    shipping_mode_road = int(shipping_mode == "Road")
    shipping_mode_sea = int(shipping_mode == "Sea")

    region_apac, region_emea = get_region_flags(origin_country, destination_country)

    # Derived flags
    heavy_shipment_flag = int(shipment_weight_kg > 1000)
    large_volume_flag = int(shipment_volume_cbm > 10)
    high_value_flag = int(shipment_value_usd > 5000)
    complex_shipment_flag = int(
        fragile_flag == 1 or temperature_control_flag == 1 or heavy_shipment_flag == 1 or large_volume_flag == 1
    )

    # Buckets (simple numeric encoding)
    if shipment_weight_kg < 100:
        weight_bucket = 0
    elif shipment_weight_kg < 1000:
        weight_bucket = 1
    else:
        weight_bucket = 2

    if shipment_volume_cbm < 1:
        volume_bucket = 0
    elif shipment_volume_cbm < 10:
        volume_bucket = 1
    else:
        volume_bucket = 2

    if shipment_value_usd < 1000:
        value_bucket = 0
    elif shipment_value_usd < 5000:
        value_bucket = 1
    else:
        value_bucket = 2

    # Carrier stats
    carrier_stats = CARRIER_DEFAULTS.get(carrier, CARRIER_DEFAULTS["Carrier_A"])
    carrier_avg_delay = carrier_stats["carrier_avg_delay"]
    carrier_sla_breach_rate = carrier_stats["carrier_sla_breach_rate"]
    carrier_volume_share = carrier_stats["carrier_volume_share"]

    # Route stats
    route = f"{origin_country}_{destination_country}"
    route_avg_delay = DEFAULT_ROUTE_AVG_DELAY
    route_sla_breach_rate = DEFAULT_ROUTE_SLA_BREACH_RATE

    # Mode stats
    mode_stats = MODE_STATS.get(shipping_mode, MODE_STATS["Unknown"])
    mode_avg_delay = mode_stats["mode_avg_delay"]
    mode_sla_breach_rate = mode_stats["mode_sla_breach_rate"]

    high_risk_route_flag = int(route_sla_breach_rate > 0.48)
    high_risk_mode_flag = int(mode_sla_breach_rate > 0.48)

    # Ratios
    cost_per_kg = shipping_cost_usd / shipment_weight_kg if shipment_weight_kg != 0 else 0
    cost_per_cbm = shipping_cost_usd / shipment_volume_cbm if shipment_volume_cbm != 0 else 0
    value_to_cost_ratio = shipment_value_usd / shipping_cost_usd if shipping_cost_usd != 0 else 0
    premium_cost_flag = int(shipping_cost_usd > 1000)

    # Risk environment
    risk_environment_encoded = 2 if (region_apac and region_emea) else 1 if (region_apac or region_emea) else 0

    # Interaction features
    fragile_and_long_route = int(fragile_flag == 1 and planned_delivery_days > 20)
    priority_but_high_delay = int(priority_flag == 1 and route_avg_delay > 0.35)
    high_cost_but_high_delay = int(shipping_cost_usd > 1000 and route_avg_delay > 0.35)
    complex_shipment_on_risky_route = int(complex_shipment_flag == 1 and high_risk_route_flag == 1)

    # Carrier one-hot
    carrier_b = int(carrier == "Carrier_B")
    carrier_c = int(carrier == "Carrier_C")
    carrier_d = int(carrier == "Carrier_D")
    carrier_e = int(carrier == "Carrier_E")
    carrier_f = int(carrier == "Carrier_F")
    carrier_g = int(carrier == "Carrier_G")
    carrier_h = int(carrier == "Carrier_H")

    return {
        "shipment_weight_kg": shipment_weight_kg,
        "shipment_volume_cbm": shipment_volume_cbm,
        "priority_flag": priority_flag,
        "fragile_flag": fragile_flag,
        "temperature_control_flag": temperature_control_flag,
        "planned_delivery_days": planned_delivery_days,
        "shipping_cost_usd": shipping_cost_usd,
        "fuel_surcharge_pct": fuel_surcharge_pct,
        "shipment_value_usd": shipment_value_usd,
        "insurance_flag": insurance_flag,
        "origin_country_encoded": origin_country_encoded,
        "destination_country_encoded": destination_country_encoded,
        "shipping_mode_Rail": shipping_mode_rail,
        "shipping_mode_Road": shipping_mode_road,
        "shipping_mode_Sea": shipping_mode_sea,
        "region_APAC": region_apac,
        "region_EMEA": region_emea,
        "heavy_shipment_flag": heavy_shipment_flag,
        "large_volume_flag": large_volume_flag,
        "high_value_flag": high_value_flag,
        "complex_shipment_flag": complex_shipment_flag,
        "weight_bucket": weight_bucket,
        "volume_bucket": volume_bucket,
        "value_bucket": value_bucket,
        "carrier_avg_delay": carrier_avg_delay,
        "carrier_sla_breach_rate": carrier_sla_breach_rate,
        "carrier_volume_share": carrier_volume_share,
        "route_avg_delay": route_avg_delay,
        "route_sla_breach_rate": route_sla_breach_rate,
        "mode_avg_delay": mode_avg_delay,
        "mode_sla_breach_rate": mode_sla_breach_rate,
        "high_risk_route_flag": high_risk_route_flag,
        "high_risk_mode_flag": high_risk_mode_flag,
        "cost_per_kg": cost_per_kg,
        "cost_per_cbm": cost_per_cbm,
        "value_to_cost_ratio": value_to_cost_ratio,
        "premium_cost_flag": premium_cost_flag,
        "risk_environment_encoded": risk_environment_encoded,
        "fragile_and_long_route": fragile_and_long_route,
        "priority_but_high_delay": priority_but_high_delay,
        "high_cost_but_high_delay": high_cost_but_high_delay,
        "complex_shipment_on_risky_route": complex_shipment_on_risky_route,
        "carrier_Carrier_B": carrier_b,
        "carrier_Carrier_C": carrier_c,
        "carrier_Carrier_D": carrier_d,
        "carrier_Carrier_E": carrier_e,
        "carrier_Carrier_F": carrier_f,
        "carrier_Carrier_G": carrier_g,
        "carrier_Carrier_H": carrier_h,
    }


def get_risk_label(probability):
    if probability < 0.30:
        return "Low Risk", "Stable"
    elif probability < 0.60:
        return "Medium Risk", "Monitor closely"
    return "High Risk", "Immediate action required"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_dict = build_feature_dict(request.form)
        input_data = [feature_dict.get(feature, 0) for feature in features]

        input_array = np.array(input_data).reshape(1, -1)
        scaled = scaler.transform(input_array)

        prediction = model.predict(scaled)[0]

        if int(prediction) == 1:
            risk_label = "High Risk"
            action_message = "Immediate action required"
        else:
            risk_label = "Low Risk"
            action_message = "Stable"

        print("Input Data:", feature_dict)
        print("Ordered Input:", input_data)
        print("Prediction:", int(prediction))

        return render_template(
            "index.html",
            prediction_text=f"SLA Breach Risk: {risk_label}",
            probability_text=None,
            action_text=f"Action: {action_message}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            probability_text=None,
            action_text=None
        )


@app.route("/test")
def test():
    sample = np.zeros(len(features)).reshape(1, -1)
    scaled = scaler.transform(sample)
    pred = model.predict(scaled)[0]
    return {"prediction": int(pred)}


if __name__ == "__main__":
    app.run(debug=True)