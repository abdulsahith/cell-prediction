from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and scaler
try:
    model1 = load_model('LNT4.h5')
    scaler = joblib.load('scaler4.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input data
        data = request.get_json()
        if not data or 'voltages' not in data or 'currents' not in data:
            return jsonify({"error": "Missing required input data"}), 400

        voltages = data['voltages']
        currents = data['currents']
        
        if len(voltages) != len(currents):
            return jsonify({"error": "Voltage and current arrays must be the same length"}), 400

        predictions = []
        for v, c in zip(voltages, currents):
            # Validate input values
            if not isinstance(v, (int, float)) or not isinstance(c, (int, float)):
                return jsonify({"error": "Invalid voltage or current value"}), 400

            # Prepare input data
            input_data = np.array([[float(v), float(c)]])
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
    app.run(debug=True, host='0.0.0.0', port=5000)