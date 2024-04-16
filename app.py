from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route("/")

def hello():
    return render_template('index.html')

@app.route("/sel_models", methods=['GET'])
def step2_sel_models():
    if 'form' in session:
        form_data = session['form']
    if 'pred_results' in session:
        pred_results_str = session['pred_results']
        pred_results = eval(pred_results_str)
    else:
        pred_results = {}
    # Model selection based on user input
    model_selection = request.args['modelSelection']
    model_path = f"{model_selection}.pkl"
    model = joblib.load(model_path)
    # Prepare the data for prediction
    features = pd.DataFrame([form_data])
    print(form_data)
    # Predict using the loaded model
    features = pd.DataFrame([form_data])  
    pipeline = joblib.load('full_pipeline.pkl')
    features_prepared = pipeline.transform(features)
    prediction = model.predict(features_prepared)
    outcome = 'Fatal' if prediction[0] == 1 else 'Non-Fatal'
    print('prediction ', prediction)
    
    pred_results[model_selection] = prediction[0]
    print('pred_results ', pred_results)
    pred_results_str = str(pred_results)
    session['pred_results'] = pred_results_str
    return render_template('/models.html', predictions=pred_results)
    #return redirect("/sel_models", code=302)

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
        
        #set session
        session['form'] = form_data
        session['pred_results'] = {}
        session.pop("pred_results", None)
        # Render result template with prediction
        return render_template('models.html', predictions={})
    except Exception as e:
        print('error')
        print(e)
        return render_template('error.html', error=str(e))  

if __name__ == "__main__":
    app.run(debug=True)
