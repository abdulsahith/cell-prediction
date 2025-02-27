from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and scaler
try:
    model_path = os.getenv("MODEL_PATH", "LNT4.h5")
    scaler_path = os.getenv("SCALER_PATH", "scaler4.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    raise
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input data
        data = request.get_json()
        if not data or 'voltages' not in data or 'currents' not in data or 'amb_temp' not in data:
            return jsonify({"error": "Missing required input data"}), 400  

        voltages = data['voltages']
        currents = data['currents']
        amb_temp = data['amb_temp']  # 

        if len(voltages) != len(currents) or len(currents) != len(amb_temp):
            return jsonify({"error": "Voltage, current, and ambient temperature arrays must be the same length"}), 400

        predictions = []
        for v, c, t in zip(voltages, currents, amb_temp):
            # Validate input values
            if not isinstance(v, (int, float)) or not isinstance(c, (int, float)) or not isinstance(t, (int, float)):
                return jsonify({"error": "Invalid voltage, current, or ambient temperature value"}), 400

            # Prepare input data
            input_data = np.array([[float(v), float(c), float(t)]])  
            input_data_normalized = scaler.transform(input_data)

            # Make prediction
            prediction = model1.predict(input_data_normalized, verbose=0)[0]

            pred_dict = {
                "top": float(round(prediction[0], 2)),
                "bottom": float(round(prediction[1], 2)),
                "left": float(round(prediction[2], 2)),
                "right": float(round(prediction[3], 2))
            }
            predictions.append(pred_dict)

        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run()
