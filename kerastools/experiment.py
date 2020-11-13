import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer

from model import Model
from misc import BatchGenerator
from evaluation import Evaluation

from tensorflow import keras

class Experiment:
    def __init__(self, config_folder, loss, optimizer, feature_length, metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.Accuracy()]):
        self.models = []
        self.callbacks= []
        self.is_categorical = False
        self.is_3d = False

        for config_file in os.listdir(config_folder):
            config_path = os.path.join(config_folder, config_file)
            model = Model.from_config(config_path, feature_length)
            if model is not None:
                self.models.append(model)

        self.compile_models(loss, optimizer, metrics)

    def compile_models(self, loss, optimizer, metrics):
        for model in self.models:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def load_data(self, X, y, val_size, test_size=None, to_categorical=False, normalize=True):
        if to_categorical:
            y = keras.utils.to_categorical(y)

        if hasattr(X[0][0], '__len__') or len(np.array(X[0]).shape) > 1:
            self.is_3d = True
            norm = Normalizer()
            X = [norm.transform(x) for x in X]
        else:
            norm = Normalizer()
            X = norm.transform(X)

        if hasattr(y[0], '__len__') or len(np.array(y).shape) > 1:
            self.is_categorical = True
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

        X_test, y_test = None, None
        if test_size is not None and test_size > 0:
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size)
            
        self.X_train, self.X_test, self.X_val = X_train, X_test, X_val
        self.y_train, self.y_test, self.y_val = y_train, y_test, y_val

    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def add_callbacks(self, callbacks):
        for cb in callbacks:
            self.add_callback(cb)

    def run_experiments(self, epochs=100, batch_size=1, shuffle=True, window_size=None, patience=None, evaluation_metrics=[classification_report], store_evaluation=None, vote_alg='hard'):
        if window_size is not None and window_size > 0 and len(np.array(self.X_train[0]).shape) < 2:
            raise Exception('X(_train, ...) must be 3-dimensional to employ windowing!')

        patience_ = []
        if patience is not None:
            patience_.append(keras.callbacks.EarlyStopping(patience=patience))

        for model in self.models:
            print('Fitting model %s !' % model)
            try:
                model.fit(BatchGenerator(self.X_train, self.y_train, batch_size=batch_size, shuffle=shuffle, window_size=window_size),
                    validation_data=BatchGenerator(self.X_val, self.y_val, window_size=window_size, shuffle=False),
                    epochs=epochs,
                    callbacks=self.callbacks + patience_)
            except:
                print('Experiment %s failed! Check your input dimensions.' % model)

        if evaluation_metrics is not None:
            for model in self.models:
                print('Evaluating %s !' % model)
                Evaluation.evaluate(model, self.X_test, self.y_test, self.is_categorical, evaluation_metrics, store_evaluation, window_size, vote_alg)
            
            

