from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd



application = Flask(__name__)
app = application

scaler = pickle.load(open('./model/standardScaler.pkl', 'rb'))
model = pickle.load(open('./model/modelForPrediction.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predictdata():
    result=""

    if request.method == 'POST':
        
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_data)


        if predict[0] == 0:
            result = "You are not diabetic"
        else:
            result = "You are diabetic"

        return render_template('single_prediction.html', result=result)
    

    else:
        return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')