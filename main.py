from flask import Flask, render_template, url_for, request
import os
from app import *

app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def root():
    if request.method == "POST":
        if request.form.get("imguri") != '':
            try:
                get_from_url(request.form.get("imguri"))
                return render_template('results.html')
            except:
                return '<h2 align=center>Forbidden Url! Cannot be processed by Url handler</h2>'
        else:
            return '<h2 align=center>Oops! bad url!</h2>'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port='8080', debug=True)