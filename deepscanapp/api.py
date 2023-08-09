# imports external libraries
import sys
import os
from os.path import exists
from datetime import date

import uvicorn
import json

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pandas as pd
from joblib import dump

# ML load custom code for train and predict
from deepscan import ds_train
from deepscan import ds_predict

# REST Models
from restdef import Trainresult
from restdef import PurReq
from restdef import Decision

print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)


# 1
# Set environnment variables for docker - only works in non dev / docker mode
UPLOAD_FOLDER = 'files'

# 2. Create app and model objects
app = FastAPI()


# 3. greetings api
#   shows the "greetings" welcom message
@app.get('/')
def index():
    return {'message': 'Greetings, Stranger got to --> /docs OR /redoc to see whats possible here'}


# 4. greetings api
#   shows a simple alive message if apllication is there
@app.get('/alive')
def alive():
    return {'message': 'alive'}


# 5. Test API to check REST-functionality of Container
#    take JSON data and return it back
@app.post('/test', response_model=PurReq)
def test_connection(PurReq: PurReq):
    data = PurReq.dict()
    return PurReq


# 6. Test API to check REST-functionality of Container
#    take JSON data and return it back
@app.get('/retrain', response_model=Trainresult)
def retrain_your_model():
    
    # pre set var if training fails
    success = False
    # call ML model training
    success = ds_train()

    # check trainings success and create return message
    if success == True:
        message = 'Training Done successfully'
    else:
        message = 'Training error please check the given data and logs'

    Trainresult = {
        "success": message,
    }
    return Trainresult


# 7. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted along with the confidence
@app.post('/predict', response_model=Decision)
def predict_anomaly(purReq: PurReq):

    # get api input as dict.
    content = purReq.dict()

    # use predict function to predict with the already trained models
    prediction, probability, confidence, prediction2, probability2, confidence2, df2 = ds_predict(content)
    
    # get id to map results
    transactionID = df2.iloc[0:1, [0]].values[0][0]


    # # Here we save our results to a daily file
    # # this can be deleted, when the response is stored externally
    today = date.today().strftime("%Y%m%d")
    filename = 'predictions_' + today + '.csv'
    basdir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(basdir, UPLOAD_FOLDER)
    path_to_file = path + '/' + filename
    # print(path_to_file)

    #if file already exist we add the results otherwise we write a new file
    file_exists = exists(path_to_file)
    if file_exists is True:
        df = pd.read_csv(path_to_file)
        dfn = pd.DataFrame({'transactionID': transactionID,
                            'prediction': prediction,
                            'probability': probability,
                            'confidence': confidence,
                            'prediction2': prediction2,
                            'probability2': probability2,
                            'confidence2': confidence2,
                            })
        
        df2 = pd.concat(df,dfn)

    else:
        df2 = pd.DataFrame({'transactionID': transactionID,
                            'prediction': prediction,
                            'probability': probability,
                            'confidence': confidence,
                            'prediction2': prediction2,
                            'probability2': probability2,
                            'confidence2': confidence2,
                            })

    #df2.to_csv(path_to_file)
    pass

    # # define api output. using mean on order items predicitons.   
    out_transactionID = transactionID
    out_prediction = np.median(prediction)
    out_probability = np.median(probability)
    out_confidence = np.median(confidence)    
    out_prediction2 = np.median(prediction2)
    out_probability2 = np.median(probability2)
    out_confidence2 = np.median(confidence2)
    

    #minimal REST response in json definition of response schema
    Decision = {
        "transactionid": out_transactionID,
        "prediction":  out_prediction,
        "probability": out_probability,
        "confidence":  out_confidence,
        "prediction2":  out_prediction2,
        "probability2": out_probability2,
        "confidence2":  out_confidence2,
    }
    
    return Decision

# 8. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, debug=True, host='0.0.0.0')
