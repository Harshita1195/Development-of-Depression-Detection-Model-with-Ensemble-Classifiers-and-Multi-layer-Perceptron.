import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sys
import os
import re
import sklearn
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from analysis import unseen_text

app = Flask(__name__)

model = pickle.load(open('mlpc.pkl', 'rb'))

@app.route('/',methods=['GET'])
@app.route('/home', methods=['GET','POST'])
def home():
        return render_template('home.html')

@app.route('/predictor', methods=['GET','POST'])
def predictor():
        if request.method == 'POST':
                text = request.form['text']
                x_test = unseen_text(text)
                no_depression = round(model.predict_proba(x_test)[0][0], 2)
                depression = round(model.predict_proba(x_test)[0][1], 2)
                return render_template('predictor.html', no_depression="Non-Depressive Sentiment: {nd}".format(nd=no_depression), depression="Depressive Sentiment: {d}".format(d=depression), txt="Entered text: '{t}'".format(t=text))
        else:
                return render_template('predictor.html')

@app.route('/work', methods=['GET'])
def work():
        return render_template('work.html')

if __name__=='__main__':
    app.run(debug=True)

