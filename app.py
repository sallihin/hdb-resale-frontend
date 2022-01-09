from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('trained_hdb_resale_estimator.pkl')

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Initialize empty list 
    features = np.empty(74)
    features.fill(0)
    month_to_buy = request.form['month_to_buy']
    year_to_buy = request.form['year_to_buy']    
    floor_area = request.form['floor_area']  
    lease_commencement_date = request.form['lease_commencement_date']  
    town = request.form['town']
    flat_type = request.form['flat_type']
    floor_number = request.form['floor_number']
    flat_model = request.form['flat_model']

    # Setting up the features array
    features[0:3] = month_to_buy, floor_area, lease_commencement_date
    features[73] = year_to_buy
    features[int(town)] = 1
    features[int(flat_type)] = 1
    features[int(floor_number)] = 1
    features[int(flat_model)] = 1
    prediction = model.predict([features])
    
    # Format prediction text for display in "index.html"
    return render_template('index.html', scroll='prediction', hdb_prediction = "Predicted price: ${:,.2f}".format(prediction[0]))   