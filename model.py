import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

#Importing train and test csv files
train_df=pd.read_csv("train.csv",nrows=2000000)
test_df=pd.read_csv("test.csv")

#functions for removing outliers
def remove_outliers_for_train(df):
    return df[(df['fare_amount'] >= 1.) & 
              (df['fare_amount'] <= 500.) &
              (df['pickup_longitude'] >= -75) & 
              (df['pickup_longitude'] <= -72) & 
              (df['dropoff_longitude'] >= -75) & 
              (df['dropoff_longitude'] <= -72) & 
              (df['pickup_latitude'] >= 40) & 
              (df['pickup_latitude'] <= 42) & 
              (df['dropoff_latitude'] >=40) & 
              (df['dropoff_latitude'] <= 42) & 
              (df['passenger_count'] >= 1) & 
              (df['passenger_count'] <= 5)]

def remove_outliers_for_test(df):
    return df[(df['pickup_longitude'] >= -75) & 
              (df['pickup_longitude'] <= -72) & 
              (df['dropoff_longitude'] >= -75) & 
              (df['dropoff_longitude'] <= -72) & 
              (df['pickup_latitude'] >= 40) & 
              (df['pickup_latitude'] <= 42) & 
              (df['dropoff_latitude'] >=40) & 
              (df['dropoff_latitude'] <= 42) & 
              (df['passenger_count'] >= 1) & 
              (df['passenger_count'] <= 5)]

#Removing outliers
train_df=remove_outliers_for_train(train_df)
test_df=remove_outliers_for_test(test_df)

#function for calculating distance
def haversine_np(lon1, lat1, lon2, lat2):
    
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

#Calculating distance(km)
train_df['distance(km)'] = haversine_np(train_df['pickup_longitude'], train_df['pickup_latitude'], train_df['dropoff_longitude'], train_df['dropoff_latitude'])
test_df['distance(km)'] = haversine_np(test_df['pickup_longitude'], test_df['pickup_latitude'], test_df['dropoff_longitude'], test_df['dropoff_latitude'])


#Training model
features = ['distance(km)','passenger_count']
label= 'fare_amount'

X=train_df[features]
y=train_df[label]

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=77)

lr_model=LinearRegression()
lr_model.fit(X_train,y_train)

#prediction
prediction=lr_model.predict(X_val)
print(prediction)

#Root Mean Squared Errror 
rmse = np.sqrt(mean_squared_error(y_val, prediction))
print("RMSE : ",rmse)

#prediction on test data set 
pre_test=lr_model.predict(test_df[features])
print(pre_test)

#making pickel file for the model
pickle.dump(lr_model,open("model.pkl","wb"))



