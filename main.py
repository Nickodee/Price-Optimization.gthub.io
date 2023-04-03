import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#load the trained model
rf_model = pickle.load(open('rf_model.pk1', 'rb'))

@app.route('/')
def home ():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = rf_model.predict(features)

    output = round(prediction[0], 2)

    return  render_template('index.html', prediction_text = 'The Optimal price is {}'.format(output))

if __name__ == "__main__":
    app.run()