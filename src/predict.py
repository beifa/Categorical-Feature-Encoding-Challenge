import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import dispatcher
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')


def predict():
    prediction = None
    df = pd.read_csv(TEST_DATA)
    idx = df['id'].values

    #loads train model, col, and encode    
    for FOLD in range(5):
        print(f'Fold : {FOLD+1}')
        df = pd.read_csv(TEST_DATA)
        le = joblib.load(f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/models/{MODEL}_{FOLD}_label_encoder.pkl')
        col = joblib.load(f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/models/{MODEL}_{FOLD}_columns.pkl')
        df = df[col]
        for c in df.columns:
            encode = le[c]
            df.loc[:, c] = encode.transform(df[c].values.tolist())    
        clf = joblib.load(f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/models/{MODEL}_{FOLD}.pkl')        
        preds = clf.predict_proba(df)[:, 1]
        if FOLD == 0:
            prediction = preds
        else:
            prediction += preds    
    prediction /= 5

    #submission (id, target)
    sub = pd.DataFrame(
        np.column_stack((idx, prediction)),
        columns=['id', 'target']
    )
    return sub



if __name__ == "__main__": 
    submit = predict()
    submit.to_csv(f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/submission/{MODEL}.csv', index = False)