from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

# File Paths
model_path = r"E:\Nirma hackthon\forntend\Try\digital_farming_model.h5"
scaler_path = r"E:\Nirma hackthon\forntend\Try\scaler.pkl"
label_encoder_path = r"E:\Nirma hackthon\forntend\Try\label_encoder.pkl"

# Load Model and Preprocessing Tools
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessing tools: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log form data
        app.logger.info(f"Form Data Received: {request.form}")

        # Get numerical values
        user_values = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['pH']),
            float(request.form['Rainfall']),
        ]

        # Get categorical values
        categorical_values = [
            request.form['Soil_Type'],
            request.form['Photoperiod'],
            request.form['Category_pH'],
            request.form['Fertility'],
            request.form['Season'],
        ]

        # One-hot encoding
        categorical_columns = ['Soil_Type', 'Photoperiod', 'Category_pH', 'Fertility', 'Season']
        encoded_features = {col: 0 for col in scaler.feature_names_in_ if col not in ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']}

        for cat, col in zip(categorical_values, categorical_columns):
            encoded_col = f"{col}_{cat}"
            if encoded_col in encoded_features:
                encoded_features[encoded_col] = 1

        # Prepare final feature set
        train_features = scaler.feature_names_in_
        input_data = np.zeros(len(train_features))

        for i, feature in enumerate(train_features[:7]):
            input_data[i] = user_values[i]

        for i, feature in enumerate(train_features[7:]):
            if feature in encoded_features:
                input_data[i + 7] = encoded_features[feature]

        # Debugging logs
        app.logger.info(f"Final Input Data: {input_data}")

        # Normalize & Predict
        sensor_data = scaler.transform([input_data])
        app.logger.info(f"Transformed Data: {sensor_data}")
        
        prediction = model.predict(sensor_data)
        predicted_class = np.argmax(prediction)
        predicted_crop = label_encoder.inverse_transform([predicted_class])[0]

        # Debugging logs
        app.logger.info(f"Raw Model Prediction: {prediction}")
        app.logger.info(f"Predicted Class Index: {predicted_class}")
        app.logger.info(f"Predicted Crop: {predicted_crop}")

        # Soil Improvement Suggestions
        recommendations = []

        if user_values[0] < 50:
            recommendations.append(f"\U0001F7E2 Increase Nitrogen (N) from {user_values[0]} to 50 using urea.")
        elif user_values[0] > 80:
            recommendations.append(f"\U0001F534 Reduce Nitrogen (N) from {user_values[0]} to below 80 to prevent over-fertilization.")

        if user_values[1] < 30:
            recommendations.append(f"\U0001F7E2 Increase Phosphorus (P) from {user_values[1]} to 30 using superphosphate.")

        if user_values[2] < 40:
            recommendations.append(f"\U0001F7E2 Increase Potassium (K) from {user_values[2]} to 40 using potash.")

        if user_values[5] < 6.5:
            recommendations.append(f"\U0001F7E2 Increase pH from {user_values[5]} to 7.0 by adding lime.")
        elif user_values[5] > 7.5:
            recommendations.append(f"\U0001F534 Reduce pH from {user_values[5]} to below 7.5 by adding sulfur.")

        return render_template("result.html", crop=predicted_crop, recommendations=recommendations)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
