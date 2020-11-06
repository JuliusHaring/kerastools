import os

from tensorflow import keras

class Model:
    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.is_fitted = False

    @staticmethod
    def from_config(config_path, feature_length, load_weights=True):
        if '.json' in os.path.splitext(config_path)[1]:
            fname = os.path.splitext(config_path)[0]

            print('Loading a config from: %s' % config_path)
            with open(config_path, 'r') as f:
                json = f.read()
                json = Model.reshape(json, feature_length)
                model = keras.models.model_from_json(json)
                
                return Model(model, config_path)
    
    @staticmethod
    def reshape(json, feature_length):
        f2d='"batch_input_shape": [null, 0]'
        f2d_fix='"batch_input_shape": [null, %i]' % feature_length
        f3d='"batch_input_shape": [null, null, 0]'
        f3d_fix='"batch_input_shape": [null, null, %i]' % feature_length

        if f2d in json:
            json = json.replace(f2d, f2d_fix)
        elif f3d in json:
            json = json.replace(f3d, f3d_fix)
        return json

    def compile(self, loss, optimizer):
        self.config.compile(loss=loss, optimizer=optimizer)

    def fit(self, generator, validation_data, epochs, callbacks):
        self.config.fit(generator, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
        self.is_fitted = True

    def get_is_fitted(self):
        return self.is_fitted
        
    def __str__(self):
        return self.name
