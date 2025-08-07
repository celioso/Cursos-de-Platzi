import joblib
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

# Cargar el modelo al iniciar la app
model = joblib.load("./models/best_models.pkl")

@app.route("/predict", methods=['GET'])
def predict():
    X_test = np.array([7.581728065,7.462271607,1.482383013,1.551121593,0.792565525,0.626006722,0.355280489,0.400770068,2.313707352])
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediccion': list(prediction)})

if __name__ == "__main__":
    app.run(port=8080)