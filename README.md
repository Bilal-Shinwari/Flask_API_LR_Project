# Flask_API_LR_Project
Linear Regression model with Flask API to Predict Taxi Fare Amount in New York City !
The Dataset is from kaggel. 
First in the Notebook i have analysis the data and have done cleaning of data.
Then i have train the model in vscode in model.py file and made a pickel file of it .
Then i have create a Flask API to do the prediction in App.py file and loaded the model prediction from model.pkl(pickel file)
The model takes to input features("distan(km)" , "passenger_count") to predict the Fare amount for Taxi .
The Flask API is hosted on localhost and i have tested the API working by Postman.
