import os
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    PATH = '/home/pka/kaggle/Categorical_Feature_Encoding_Challenge/input/'
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=13)
    for fold, (t_idx, v_idx) in enumerate(kf.split(X=df, y= df.target.values)):
        print(len(t_idx), len(v_idx))
        df.loc[v_idx, 'kfold'] = fold
    df.to_csv(os.path.join(PATH, 'train_folds.csv'), index = False)

