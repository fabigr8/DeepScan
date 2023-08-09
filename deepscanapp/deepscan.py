import platform
#import sys
import os
import numpy as np
#import scipy
import pandas as pd
import json
from pandas import json_normalize
from datetime import datetime as dt, timedelta
from joblib import dump, load

#preprocessing bibs
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
#from sklearn import preprocessing
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split



#ML-Models bibs
#dependencies
#import tensorflow as tf
#import keras as ks
#from tensorflow.keras import layers
#from tensorflow.keras import losses

#load models to predict
#clf_copod = load(MODEL_FILE_COPOD)
#prediction, probability, confidence = clf_copod.predict(data)

#clf_if = load(MODEL_FILE_IF)
#prediction2, probability2, confidence2 = clf_if.predict(data)



# import algo bibs
import pyod as pyod
from pyod.models.copod import COPOD
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest

#print(tf.keras.__version__)
#print(ks.__version__)

def ds_train_test():

    return True

# ####################### Train
#Train the model
def ds_train():

    # paths 4 persisting model
    MODEL_DIR = os.environ["MODEL_DIR"]

    MODEL_FILE_COPOD = os.environ["MODEL_FILE_COPOD"]
    MODEL_FILE_IF = os.environ["MODEL_FILE_IF"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_FILE_PIPEL = os.environ["MODEL_FILE_PIPEL"]

    MODEL_PATH_COPOD = os.path.join(MODEL_DIR, MODEL_FILE_COPOD)
    MODEL_PATH_IF = os.path.join(MODEL_DIR, MODEL_FILE_IF)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    MODEL_PATH_PIPEL = os.path.join(MODEL_DIR, MODEL_FILE_PIPEL)

    # load data from json
    with open('response_weclapp_purchase_order.json', encoding='utf-8') as f:
        data3 = json.load(f)

    print(type(data3))

    df = pd.json_normalize(
        data3,
        record_path=['result'],
        errors='ignore'
    ).explode("purchaseOrderItems").explode("statusHistory")


    df2 = pd.concat([df, df.pop("purchaseOrderItems").apply(pd.Series).add_prefix('item_'), df.pop("statusHistory").apply(pd.Series).add_prefix('OrderStat_')], axis=1)

    # drop empty cols
    df2.drop(['customAttributes','purchaseOrderNumber',
          'shippingCostItems','tags',
          'item_customAttributes','item_description',
          'item_reductionAdditionItems', 'warehouseName'],axis=1, inplace=True)
    
    # FIX data types 
    l_cols = set(df2.columns.values.tolist())
    nums = ['grossAmount', 
            'grossAmountInCompanyCurrency', 
            'netAmountInCompanyCurrency', 
            'item_grossAmount', 
            'item_grossAmountInCompanyCurrency',
            'item_netAmount',
            'item_netAmountForStatistics',
            'item_netAmountForStatisticsInCompanyCurrency',
            'item_netAmountInCompanyCurrency',
            'item_quantity',
            'item_receivedQuantity',
            'item_unitPrice',
            'item_unitPriceInCompanyCurrency']

    nums = set(nums)


    tmstps = ['lastModifiedDate',
                'createdDate',
                'orderDate',
                'plannedDeliveryDate',
                'servicePeriodFrom',
                'servicePeriodTo',
                'OrderStat_statusDate',
                'item_lastModifiedDate',
                'item_createdDate']
    
    tmstps = set(tmstps)

    cats = l_cols - nums
    cats = cats - tmstps

    num_feats = list(nums).copy()
    tms_feats = list(tmstps).copy()
    cat_feats = list(cats).copy()

    # date feats to dates
    for col in tms_feats:
        df2[col] = df2[col].fillna(0)
        df2[col] = df2[col].astype(float) / 1000
        df2[col] = df2[col].apply(lambda x: dt.fromtimestamp(x))

    # numerical Feats to numerical Feats
    for col in num_feats:
        df2[col] = df2[col].astype('float64')

    # Cat Feats to Cat Feats
    for col in cat_feats:
        df2[col] = df2[col].astype('category')

    # PIPELINE as preprocessing 
    #create a pipeline for numerical vars 
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        #('std_scaler', Normalizer(norm='l2')),
    ])
    
    #create a pipeline for categorical vars 
    cat_pipeline = Pipeline([
        #('imputer', SimpleImputer(strategy="median")),
        ('OHE',OneHotEncoder())
    ])

    #create a pipeline for categorical vars 
    dat_pipeline = Pipeline([
        ('date_calcu', DateTransformer()), #
    ])

    #creat feature resorts
    num_attribs = num_feats.copy()
    cat_attribs = cat_feats.copy()
    dat_attribs = tms_feats.copy()

    # full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("dat", dat_pipeline, dat_attribs),
    ])

    # use pipeline
    df_prepared = full_pipeline.fit_transform(df2)   


    ### Data Split
    df_train, df_test = train_test_split(df_prepared, test_size=0.2, random_state=42)

    ### training
    #parameters created via RepeatedStratifiedKFold

    contamination = 0.025  # specify outlier perc. if known
    n_features = len(df_train)
    n_jobs = 5 # specify cores used

    ### first OD-A
    # clf_name = 'COPOD'
    clf_copod = COPOD(contamination=contamination, n_jobs=n_jobs)
    clf_copod.fit(df_train)


    # get outlier scores / labels / confidence
    y_train_scores = clf_copod.decision_scores_  # outlier scores on train
    y_train_pred = clf_copod.labels_  # outlier Labels on trainclf_copod
    y_train_pred_confidence = clf_copod.predict_confidence(df_train) # outlier confidence ont train

    y_test_scores = clf_copod.decision_function(df_test)  # predicted outlier scores on test
    y_test_pred, y_test_pred_confidence = clf_copod.predict(df_test, return_confidence=True) # predicted outlier labels and conf on test

    
    #define  empty dataframes df_erg_test and df_erg_train with given colums for results
    cols=['copod_preds','copod_scores','copod_conf', 'AE_Keras_preds', 'AE_Keras_scores', 'AE_Keras_conf']
    df_erg_test = pd.DataFrame(columns=cols)
    df_erg_train = pd.DataFrame(columns=cols)

   
    #we save our train and test results in two df to export and check
    df_erg_test['copod_preds'] = y_test_pred.tolist()
    df_erg_test['copod_scores'] = y_test_scores.tolist()
    df_erg_test['copod_conf'] = y_test_pred_confidence.tolist()
    df_erg_train['copod_preds'] = y_train_pred.tolist()
    df_erg_train['copod_scores'] = y_train_scores.tolist()
    df_erg_train['copod_conf'] = y_train_pred_confidence.tolist()


    from pyod.models.iforest import IForest
    # removed hence performance of test machine 
    ### second our KERAS AE
    # train AutoEncoder
    # train AutoEncoder detector
    #clf_nn_name = 'AutoEncoder'
    #clf_nn = AutoEncoder(
    #    hidden_neurons=[93, 58, 16, 58, 93],
    #    hidden_activation='relu',
    #    output_activation='sigmoid',
    #    #loss=keras.losses.mean_squared_error,
    #    optimizer='adam',
    #    epochs=60,
    #    batch_size=32,
    #    dropout_rate=0.2,
    #    l2_regularizer=0.1,
    #    validation_size=0.1,
    #    preprocessing=True,
    #    random_state=None,
    #    contamination=contamination)

    #clf_nn.fit(train)

    # clf_if_name = 'IForest'
    clf_if = IForest(contamination=contamination, n_jobs=n_jobs)
    clf_if.fit(df_train)

    # get outlier scores / labels / confidence
    y_train_scores_ae = clf_if.decision_scores_  # outlier scores on train
    y_train_pred_ae = clf_if.labels_  # outlier Labels on train
    y_train_pred_confidence_ae = clf_if.predict_confidence(df_train) # outlier confidence ont train

    y_test_scores_ae = clf_if.decision_function(df_test)  # predicted outlier scores on test
    y_test_pred_ae, y_test_pred_confidence_ae = clf_if.predict(df_test, return_confidence=True) # predicted outlier labels and conf on test

    #we save our train and test results in two df to export and check
    df_erg_train['AE_Keras_preds'] = y_train_pred_ae.tolist()
    df_erg_train['AE_Keras_scores'] = y_train_scores_ae.tolist()
    df_erg_train['AE_Keras_conf'] = y_train_pred_confidence_ae.tolist()

    df_erg_test['AE_Keras_preds'] = y_test_pred_ae.tolist()
    df_erg_test['AE_Keras_scores'] = y_test_scores_ae.tolist()
    df_erg_test['AE_Keras_conf'] = y_test_pred_confidence_ae.tolist()

    ### export and save everything
    # Serialize model and pipeline
    dump(clf_copod, MODEL_PATH_COPOD)
    dump(clf_if, MODEL_PATH_IF)
    dump(full_pipeline, MODEL_PATH_PIPEL)

    df_erg_test.to_csv('./test_erg.csv')
    df_erg_train.to_csv('./train_erg.csv')

    return True

# ####################### PREDICT
#load pipeline, model, and predict
def ds_predict(data):

    # paths 4 persisting model
    MODEL_DIR = os.environ["MODEL_DIR"]

    MODEL_FILE_COPOD = os.environ["MODEL_FILE_COPOD"]
    MODEL_FILE_IF = os.environ["MODEL_FILE_IF"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_FILE_PIPEL = os.environ["MODEL_FILE_PIPEL"]

    MODEL_PATH_COPOD = os.path.join(MODEL_DIR, MODEL_FILE_COPOD)
    MODEL_PATH_IF = os.path.join(MODEL_DIR, MODEL_FILE_IF)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    MODEL_PATH_PIPEL = os.path.join(MODEL_DIR, MODEL_FILE_PIPEL)
    
    # load data from json
    # with open(jsond, encoding='utf-8') as f:
    #     data3 = json.load(f)
    data3 = data
    print(type(data3))

    df = pd.json_normalize(
        data3,
        record_path=['result'],
        errors='ignore'
    ).explode("purchaseOrderItems").explode("statusHistory")


    df2 = pd.concat([df, df.pop("purchaseOrderItems").apply(pd.Series).add_prefix('item_'), df.pop("statusHistory").apply(pd.Series).add_prefix('OrderStat_')], axis=1)

    # drop empty cols
    df2.drop(['customAttributes','purchaseOrderNumber',
          'shippingCostItems','tags',
          'item_customAttributes','item_description',
          'item_reductionAdditionItems', 'warehouseName'],axis=1, inplace=True)
    
    # FIX data types 
    l_cols = set(df2.columns.values.tolist())
    nums = ['grossAmount', 
            'grossAmountInCompanyCurrency', 
            'netAmountInCompanyCurrency', 
            'item_grossAmount', 
            'item_grossAmountInCompanyCurrency',
            'item_netAmount',
            'item_netAmountForStatistics',
            'item_netAmountForStatisticsInCompanyCurrency',
            'item_netAmountInCompanyCurrency',
            'item_quantity',
            'item_receivedQuantity',
            'item_unitPrice',
            'item_unitPriceInCompanyCurrency']

    nums = set(nums)


    tmstps = ['lastModifiedDate',
                'createdDate',
                'orderDate',
                'plannedDeliveryDate',
                'servicePeriodFrom',
                'servicePeriodTo',
                'OrderStat_statusDate',
                'item_lastModifiedDate',
                'item_createdDate']
    
    tmstps = set(tmstps)

    cats = l_cols - nums
    cats = cats - tmstps

    num_feats = list(nums).copy()
    tms_feats = list(tmstps).copy()
    cat_feats = list(cats).copy()

    # date feats to dates
    for col in tms_feats:
        df2[col] = df2[col].fillna(0)
        df2[col] = df2[col].astype(float) / 1000
        df2[col] = df2[col].apply(lambda x: dt.fromtimestamp(x))


    # # use pipeline
    # # load the pipeline from disk
    full_pipeline = load(MODEL_PATH_PIPEL)
    df_preped = full_pipeline.transform(df2)

    
    # ## predict
    # load copod model
    clf_copod = load(MODEL_PATH_COPOD)
    # predicted outlier scores on test
    scores = clf_copod.decision_function(df_preped) 
    # predicted outlier labels and conf on test
    pred, confidence = clf_copod.predict(df_preped, return_confidence=True)

    # ## predict
    # load copod model
    clf_if = load(MODEL_PATH_IF)
    # predicted outlier scores on test
    scores2 = clf_if.decision_function(df_preped) 
    # predicted outlier labels and conf on test
    pred2, confidence2 = clf_if.predict(df_preped, return_confidence=True)

    return pred, scores, confidence, pred2, scores2, confidence2, df2


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        #test 
        return None
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tmstps = ['lastModifiedDate','createdDate', 'orderDate','plannedDeliveryDate','servicePeriodFrom','servicePeriodTo', 
                  'OrderStat_statusDate','item_lastModifiedDate', 'item_createdDate']
        
        X['date_diff1'] = (X['lastModifiedDate'] - X['createdDate']).dt.total_seconds() /3600
        X['date_diff2'] = (X['orderDate'] - X['plannedDeliveryDate']).dt.total_seconds() /3600
        X['date_diff3'] = (X['servicePeriodTo'] - X['servicePeriodFrom']).dt.total_seconds() /3600
        X['date_diff4'] = (X['OrderStat_statusDate'] - X['createdDate']).dt.total_seconds() /3600
        X['date_diff5'] = (X['item_lastModifiedDate'] - X['item_createdDate']).dt.total_seconds() /3600
        X.drop(tmstps, axis=1,inplace=True)
        return X      
    



if __name__ == '__main__':
    train()




        