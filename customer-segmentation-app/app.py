from flask import Flask, request, jsonify
from urllib.parse import quote as url_quote
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model using pickle
model_path = './model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def hello_world():
   return "Hello world!"

@app.route('/predict', methods=['POST'])
def predict():
    print("Inside Prediction")
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
