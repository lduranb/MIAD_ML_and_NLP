#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_price(url):
 
    reg = joblib.load('xgb.pkl') 

    url_ = pd.DataFrame([url], columns=['car'])

    model_cat = pd.read_csv('modelc.csv')
    make_cat = pd.read_csv('makec.csv')
    state_cat = pd.read_csv('statec.csv')
    trc_cat = pd.read_csv('trcc.csv')
    test_cols = pd.read_csv('testcol.csv')

    # Create features  
    veh = url_.car.str.split(' ', expand=True)
    url_["Year"] = int(veh.iloc[:, 0])
    url_["Mileage"] = int(veh.iloc[:, 1])
    url_["State"] = " "+veh.iloc[:, 2]
    url_["Make"] = veh.iloc[:, 3]
    url_["Model"] = veh.iloc[:, 4]
    url_["TRC"] = [i[-3:] for i in url_["Model"]]
    url_["Age"] = 2019 - int(url_["Year"])

    url_ = url_.join(model_cat.set_index("Model"), on="Model")
    url_ = url_.join(make_cat.set_index("Make"), on="Make")
    url_ = url_.join(state_cat.set_index("State"), on="State")
    url_ = url_.join(trc_cat.set_index("TRC"), on="TRC")

    dumm_cols = ["Model_range","Model_median","Make_range","Make_median","State_range","State_median",\
                               "TRC_range","TRC_median"]

    dumm = pd.DataFrame({i+"_"+str(int(url_[i][0])):[1] for i in dumm_cols})
    url_ = url_.join(dumm)
    cols = url_.select_dtypes(include=['object']).columns
    url_.drop(cols, axis=1, inplace=True)
    url_.drop(dumm_cols, axis=1, inplace=True)
    url_ = pd.concat([test_cols,url_]).fillna(0)
    url_ = url_.iloc[1:]

    # Make prediction
    p1 = reg.predict(url_)

    return p1[0]


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_price(url)
        
        print(url)
        print('Price prediction: ', p1[0])
        