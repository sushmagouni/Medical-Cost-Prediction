from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model once when the app starts
with open('rf.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from form datails
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Encode categorical variables
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0
        region_encoded = {
            'northeast': 0,
            'northwest': 1,
            'southeast': 2,
            'southwest': 3
        }.get(region, -1)

        # Ensure region is valid
        if region_encoded == -1:
            return render_template('predict.html', pred='Invalid region specified!')

        # Create feature array
        features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

        # Predict using the model
        prediction = model.predict(features)[0]

        if prediction < 0:
            return render_template('predict.html', prediction='Error calculating Amount!')
        else:
            return render_template('predict.html', prediction='Expected amount is {0:.3f}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
