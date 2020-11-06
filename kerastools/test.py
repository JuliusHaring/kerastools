from model import Model
from experiment import Experiment
from sklearn.metrics import roc_auc_score, classification_report

import numpy as np

from tensorflow import keras
fl = 10
exp = Experiment('./configs/', loss='categorical_crossentropy', optimizer= keras.optimizers.Nadam(), feature_length=fl)

n = 1000
exp.load_data(np.random.rand(n, fl), np.random.randint(0,2,size=n), to_categorical=True, val_size=0.2, test_size=0.5)

exp.run_experiments(patience=2, epochs=10,batch_size=256 , evaluation_metrics=[roc_auc_score, classification_report], store_evaluation='./configs/stats/')