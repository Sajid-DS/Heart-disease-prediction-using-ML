# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 03:26:15 2021

@author: hp
"""
from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
import requests
app = Flask(__name__)
loaded = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
# prediction function

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
           print(request.form.get('age'))
           print(request.form.get('sex'))
           print(request.form.get('cp'))
           print(request.form.get('trestbps'))
           print(request.form.get('chol'))
           print(request.form.get('fbs'))
           print(request.form.get('restecg'))
           print(request.form.get('thalach'))
           print(request.form.get('exang'))
           print(request.form.get('oldpeak'))
           print(request.form.get('slope'))
           print(request.form.get('ca'))
           print(request.form.get('thal'))
           
           age=int(request.form['age'])
           sex=int(request.form['sex'])
           cp=int(request.form['cp'])
           trestbps=int(request.form['trestbps'])
           chol=int(request.form['chol'])
           fbs=int(request.form['fbs'])
           restecg=int(request.form['restecg'])
           thalach=int(request.form['thalach'])
           exang=int(request.form['exang'])
           oldpeak=float(request.form['oldpeak'])
           slope=int(request.form['slope'])
           ca=int(request.form['ca'])
           thal=int(request.form['thal'])
           to_predict_list = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
           #to_predict = np.array(to_predict_list).reshape(1,7)
           #int_features = [int(x) for x in request.form.values()]
           final_features = np.array(to_predict_list).reshape(1,-1)
           result = loaded.predict(final_features)	
           if result[0]==0: 
	           return render_template("index.html",prediction="Person Don't have Heart Disease ")
           else:
               return render_template("index.html",prediction="Person is suffering from Heart Disease ")
               
if __name__=='__main__':
     app.run(debug=True)