import numpy as np
import pandas as pd
import pickle
from flask import Flask,render_template,request,jsonify

#Creating flask app
app=Flask(__name__)

#Loading pickle file
model=pickle.load(open("model.pkl","rb"))

#Rest API
@app.route("/predict",methods=["POST"])
def predict():
    json_=request.json
    df=pd.DataFrame(json_)
    prediction=model.predict(df)
    return jsonify({"Prediction":list(prediction)})

if __name__=="__main__":
    app.run(debug=True)