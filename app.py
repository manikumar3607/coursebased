from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your dataset
data = pd.read_csv("C:/Users/manil/OneDrive/Desktop/emeralds1/deploy-ml-model-flask-master/basic.csv")

# Select relevant features (volume and fuel type) and the target variable (CO2 emissions)
X = data[["Volume", "Fuel"]]
y = data["CO2e (mln tons)"]

# Convert fuel type to numerical labels (assuming it's categorical)
label_encoder = LabelEncoder()
X["Fuel"] = label_encoder.fit_transform(X["Fuel"])

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X, y)

# Route for your homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the request
    data = request.json
    volume = float(data['volume'])
    fuel = data['fuel']
    fuel_encoded = label_encoder.transform([fuel])[0]

    # Make prediction
    prediction = rf_model.predict([[volume, fuel_encoded]])

    # Return prediction
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)