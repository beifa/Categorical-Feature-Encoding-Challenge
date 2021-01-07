from sklearn import ensemble

MODELS = {
    'rf': ensemble.RandomForestClassifier(n_jobs=-1, verbose = 2),
    'extra_tree': ensemble.ExtraTreesClassifier(n_jobs=-1, verbose = 2)
}