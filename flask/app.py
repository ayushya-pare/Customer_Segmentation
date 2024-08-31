import base64
from io import BytesIO
import pandas as pd
import seaborn as sns
from flask import Flask
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
#model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Customer Segmentation and Churn Prediction API!"

'''
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())
'''
@app.route('/plot')
def plot():
    plt.figure(figsize=(10,6))
    sns.countplot(x='Churn', data=df)
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig = plt.gcf()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == '__main__':
    df = pd.read_csv('telecom_users.csv')
    #df = df.astype(int)
    app.run(host='0.0.0.0', port=5000, debug=True)
