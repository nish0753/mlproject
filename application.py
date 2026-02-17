import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Enable CORS for all routes
CORS(app)

# route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=request.form.get('writing_score'),
            reading_score=request.form.get('reading_score'),
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Pass the input data and the rounded result to the template
        return render_template('home.html', results=round(results[0], 2), data=data)
    else:
        return render_template('home.html')

# New JSON API endpoint for React frontend
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=request.form.get('writing_score'),
            reading_score=request.form.get('reading_score'),
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return jsonify({
            'success': True,
            'predicted_score': round(results[0], 2),
            'message': 'Prediction completed successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to make prediction'
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)