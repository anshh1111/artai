from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Ensure the model file exists
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ERROR: {model_path} not found. Ensure it is in the same directory as app.py.")

# Load the trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Validate input
    if "features" not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400
    
    try:
        input_features = np.array(data["features"]).reshape(1, -1)  # Convert to array
        prediction = model.predict(input_features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
