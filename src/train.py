import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import dispatcher
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')


FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}


if __name__ == "__main__":    
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA) 

    """
    example fold = 2
    train_df = FOLD_MAPPING.get(2) --> [0,1,3,4]
    valid_df = fold 2   
    
    """    
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis = 1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis = 1)

    #check/make correct columns train and valid
    valid_df = valid_df[train_df.columns]

    #encode
    label_encode = {}
    
    for c in train_df.columns:
        le = preprocessing.LabelEncoder()
        le.fit(
            train_df[c].values.tolist() +
            valid_df[c].values.tolist() +
            df_test[c].values.tolist()
            )       
        train_df.loc[:, c] = le.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = le.transform(valid_df[c].values.tolist())
        label_encode[c] = le


    # train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print('POC_AUC :', metrics.roc_auc_score(yvalid, preds))
    print('Save')
    joblib.dump(label_encode, f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/models/{MODEL}_{FOLD}_label_encoder.pkl')
    joblib.dump(clf, f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/models/{MODEL}_{FOLD}.pkl')
    joblib.dump(train_df.columns, f'/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/models/{MODEL}_{FOLD}_columns.pkl')
    