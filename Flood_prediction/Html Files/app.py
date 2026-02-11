from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os

app = Flask(__name__)

# ---------- SAFE MODEL LOADING ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'floods.save')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'transform.save')

model = None
sc = None

def load_artifacts():
    global model, sc
    try:
        model = load(MODEL_PATH)
        sc = load(SCALER_PATH)
        print("✅ Model and Scaler loaded successfully")
    except Exception as e:
        print("❌ Error loading model/scaler")
        print(e)
        model = None
        sc = None

load_artifacts()
# ----------------------------------------


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('index.html')


@app.route('/data_predict', methods=['POST'])
def data_predict():

    # SAFETY CHECK
    if model is None or sc is None:
        return "Model or Scaler not loaded. Check floods.save / transform.save files."

    try:
        # MATCHING WITH index.html NAMES
        cloud = float(request.form['cloud'])
        rainfall = float(request.form['rainfall'])
        jan_feb = float(request.form['jan_feb'])
        mar_may = float(request.form['mar_may'])
        jun_sep = float(request.form['jun_sep'])

        data = np.array([[cloud, rainfall, jan_feb, mar_may, jun_sep]])
        data_scaled = sc.transform(data)

        prediction = model.predict(data_scaled)[0]

        if prediction == 1:
            return render_template('chance.html')
        else:
            return render_template('nochance.html')

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
