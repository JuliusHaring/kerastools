from tensorflow import keras
import numpy as np

class BatchGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True, window_size=None):
        'Initialization'
        X_temp, y_temp, indices = [],[], []
        if window_size is not None and window_size > 1:
            for idx, (X_, y_) in enumerate(zip(X, y)):
                for i in range(len(X_)-window_size):
                    X_temp.append(X_[i:i+window_size])
                    y_temp.append(y_)
                    indices.append(idx)
            X, y = X_temp, y_temp

        self.indices = indices
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def get_voting_help(self):
        return self.indices, self.y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb