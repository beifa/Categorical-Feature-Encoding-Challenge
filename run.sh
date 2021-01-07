export TRAINING_DATA=/home/pka/Categorical_Feature_Encoding_Challenge/input/train_folds.csv
export TEST_DATA=/home/pka/Categorical_Feature_Encoding_Challenge/input/test.csv
export MODEL=$1

# FOLD=0 python3 src/train.py
# FOLD=1 python3 src/train.py
# FOLD=2 python3 src/train.py
# FOLD=3 python3 src/train.py
# FOLD=4 python3 src/train.py

python3 src/predict.py
