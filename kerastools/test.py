from model import Model
from experiment import Experiment
from sklearn.metrics import roc_auc_score, classification_report

import numpy as np

from tensorflow import keras
fl = 10
exp = Experiment('./configs/', loss='categorical_crossentropy', optimizer= keras.optimizers.Nadam(lr=0.05), feature_length=fl)

n = 1000
features = []
for i in range(n):
    f = [0,0,0,0,0,0,0,0,0,0]
    f[np.random.randint(0,10)] = 1
    features.append(f)
features = np.array(features)

exp.load_data(features, features, val_size=0.2, test_size=0.5)

exp.run_experiments(epochs=100, patience=10,batch_size=10, evaluation_metrics=[classification_report], store_evaluation='./configs/stats/')