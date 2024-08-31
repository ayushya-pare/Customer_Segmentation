import base64
from io import BytesIO
import seaborn as sns
import pandas as pd
import pickle
from flask import Flask, jsonify
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model pipeline
model_path = './model.pkl'
with open(model_path, 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Load the feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def hello_world():
    return "Welcome to the Customer Segmentation and Churn Prediction API!"

@app.route('/plot')
def plot():
    sns.countplot(x='Churn', data=df)
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig = plt.gcf()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

@app.route('/predict', methods=['GET'])
def predict():
    # Use the loaded dataframe to make predictions
    input_data = df.drop('Churn', axis=1)
    
    # Ensure the input data has the same columns as during training
    input_data = input_data[feature_names]
    
    predictions = pipeline.predict(input_data)
    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == "__main__":
    df = pd.read_csv('telecom_users.csv')
    app.run(host='0.0.0.0', port=5001, debug=True)
