import os
import numpy as np

from sklearn.model_selection import train_test_split

from model import Model
from misc import BatchGenerator
from evaluation import Evaluation

from tensorflow import keras

class Experiment:
    def __init__(self, config_folder, loss, optimizer, feature_length):
        self.models = []
        self.callbacks= []
        self.to_categorical = False

        for config_file in os.listdir(config_folder):
            config_path = os.path.join(config_folder, config_file)
            model = Model.from_config(config_path, feature_length)
            if model is not None:
                self.models.append(model)

        self.compile_models(loss, optimizer)

    def compile_models(self, loss, optimizer):
        for model in self.models:
            model.compile(loss=loss, optimizer=optimizer)

    def load_data(self, X, y, val_size, test_size=None, to_categorical=False):
        if to_categorical:
            y = keras.utils.to_categorical(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

        X_test, y_test = None, None
        if test_size is not None and test_size > 0:
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size)
            
        self.X_train, self.X_test, self.X_val = X_train, X_test, X_val
        self.y_train, self.y_test, self.y_val = y_train, y_test, y_val

        self.to_categorical = to_categorical

    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def add_callbacks(self, callbacks):
        for cb in callbacks:
            self.add_callback(cb)

    def run_experiments(self, epochs=100, batch_size=32, shuffle=True, window_size=None, patience=None, evaluate=True):
        patience_ = []
        if patience is not None:
            patience_.append(keras.callbacks.EarlyStopping(patience=patience))

        for model in self.models:
            try:
                model.fit(BatchGenerator(self.X_train, self.y_train, batch_size=batch_size, shuffle=shuffle, window_size=window_size),
                    validation_data=BatchGenerator(self.X_val, self.y_val, window_size=window_size, shuffle=False),
                    epochs=epochs,
                    callbacks=self.callbacks + patience_)
            except:
                print('Experiment %s failed!' % model)

        if evaluate:
            for model in self.models:
                if model.get_is_fitted():
                    print('Evaluating %s!' % model)
                    Evaluation.evaluate(model, self.X_test, self.y_test, self.to_categorical)
            
            

