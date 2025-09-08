from flask import Flask, request, render_template
import joblib
import numpy as np

# Load model
model = joblib.load("crop_recommender.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input
        sample_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(sample_input)

        return render_template('index.html', result=prediction[0])
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
