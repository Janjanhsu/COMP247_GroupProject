from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/result", methods=['POST'])
def result():
    try:
        # Collect all form data
        form_data = {
            'accloc': request.form['accloc'],
            'drivact': request.form['drivact'],
            'day': int(request.form['day']),
            'district': request.form['district'],
            'speeding': int(request.form['speeding']),
            'invtype': request.form['invtype'],
            'division': request.form['division'],
            'pedestrian': int(request.form['pedestrian']),
            'vehtype': request.form['vehtype'],
            'drivcond': request.form['drivcond'],
            'truck': int(request.form['truck']),
            'impactype': request.form['impactype'],
            'latitude': float(request.form['latitude']),
            'longitude': float(request.form['longitude']),
            'invage': float(request.form.get('invage', 0))  
        }

        # Model selection based on user input
        model_selection = request.form['modelSelection']
        model_path = f"{model_selection}.pkl"  
        model = joblib.load(model_path)

        # Prepare the data for prediction
        features = pd.DataFrame([form_data])  

        # Predict using the loaded model
        prediction = model.predict(features)
        outcome = 'Fatal' if prediction[0] == 1 else 'Non-Fatal'
        
        # Render result template with prediction
        return render_template('result.html', prediction=outcome)
    except Exception as e:
        return render_template('error.html', error=str(e))  

if __name__ == "__main__":
    app.run(debug=True)
