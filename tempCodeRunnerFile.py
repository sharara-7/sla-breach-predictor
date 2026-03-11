import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load files using absolute path
model = joblib.load(os.path.join(BASE_DIR, "sla_svm_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.form

    input_data = [float(data.get(feature, 0)) for feature in features]

    input_array = np.array(input_data).reshape(1, -1)

    scaled = scaler.transform(input_array)

    prediction = model.predict(scaled)[0]

    if prediction == 1:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return render_template("index.html", prediction_text=f"SLA Breach Risk: {risk}")
    
@app.route("/test")
def test():
    sample = [0] * len(features)

    arr = np.array(sample).reshape(1, -1)
    scaled = scaler.transform(arr)

    pred = model.predict(scaled)[0]

    return {"prediction": int(pred)}

@app.route("/ui")
def ui():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)