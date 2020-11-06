from model import Model
from experiment import Experiment

import numpy as np

from tensorflow import keras

exp = Experiment('./configs/', loss='categorical_crossentropy', optimizer= keras.optimizers.Nadam(), feature_length=21)

exp.load_data(np.random.rand(100,21), np.random.rand(100, 2), val_size=0.2, test_size=0.2)

exp.run_experiments(patience=2, epochs=10)