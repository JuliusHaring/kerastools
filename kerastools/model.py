import os

from tensorflow import keras

class Model:
    def __init__(self, config, name):
        self.config = config
        self.name = os.path.splitext(name)[0]
        self.is_fitted = False

    @staticmethod
    def from_config(config_path, feature_length=None, custom_objects=None, load_weights=True):
        extension = os.path.splitext(config_path)[1]
        if '.json' in extension:
            if feature_length is None:
                raise Exception('\'feature_length\' must be provided if .json config files are provided.')
            fname = os.path.splitext(config_path)[0]
            print('Loading a config from: %s' % config_path)

            with open(config_path, 'r') as f:
                json = f.read()
                json = Model.reshape(json, feature_length)
                model = keras.models.model_from_json(json)
                
                return Model(model, config_path)
        if '.h5' in extension:
            fname = os.path.splitext(config_path)[0]
            try:
                print('Loading a config from: %s' % config_path)

                model = keras.models.load_model(config_path, custom_objects=custom_objects)
                return Model(model, config_path)
            except:
                print('Tried to load a .h5 file that is not a model. Skipping %s' % fname)

    
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

    def compile(self, loss, optimizer, metrics):
        self.config.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, generator, validation_data, epochs, callbacks):
        self.config.fit(generator, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
        self.is_fitted = True

    def predict(self, X):
        return self.config.predict(X)

    def get_is_fitted(self):
        return self.is_fitted
        
    def __str__(self):
        return self.name
