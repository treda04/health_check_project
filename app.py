# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'O.\x89\xcc\xa0>\x96\xf7\x871\xa2\xe6\x9a\xe4\x14\x91\x0e\xe5)\xd9'

# Disable caching for static files during development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load the Random Forest Classifier model
filename = 'Models/diabetes-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('d_result.html', prediction=my_prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('diabetes'))


def ValuePredictor(to_predict_list, size):
    loaded_model = joblib.load('models/heart_model')
    to_predict = np.array(to_predict_list).reshape(1, size)
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list, 11)

            if(int(result) == 1):
                prediction = 1
            else:
                prediction = 0

            return render_template('h_result.html', prediction=prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('heart'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

if __name__ == '__main__':
    app.run(debug=True)
