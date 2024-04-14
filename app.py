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
            'ACCLOC': request.form['accloc'],
            'DRIVACT': request.form['drivact'],
            'DAY': int(request.form['day']),
            'DISTRICT': request.form['district'],
            'SPEEDING': int(request.form['speeding']),
            'INVTYPE': request.form['invtype'],
            'DIVISION': request.form['division'],
            'PEDESTRIAN': int(request.form['pedestrian']),
            'VEHTYPE': request.form['vehtype'],
            'DRIVCOND': request.form['drivcond'],
            'TRUCK': int(request.form['truck']),
            'IMPACTYPE': request.form['impactype'],
            'LATITUDE': float(request.form['latitude']),
            'LONGITUDE': float(request.form['longitude']),
            'INVAGE': float(request.form.get('invage', 0))  
        }

        # Model selection based on user input
        model_selection = request.form['modelSelection']
        model_path = f"{model_selection}.pkl"  
        model = joblib.load(model_path)

        # Prepare the data for prediction
        features = pd.DataFrame([form_data])  
        pipeline = joblib.load('full_pipeline.pkl')
        features_prepared = pipeline.transform(features)
        # Predict using the loaded model
        prediction = model.predict(features_prepared)
        outcome = 'Fatal' if prediction[0] == 1 else 'Non-Fatal'
        
        # Render result template with prediction
        return render_template('result.html', prediction=outcome)
    except Exception as e:
        return render_template('error.html', error=str(e))  

if __name__ == "__main__":
    app.run(debug=True)
