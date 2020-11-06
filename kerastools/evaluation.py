import re
import os
from collections import Counter
import numpy as np
from datetime import date

from misc import get_separator, BatchGenerator

class Evaluation:
    @staticmethod
    def vote(model, X, y, vote_alg, window_size):
        generator = BatchGenerator(X, y, window_size=window_size, shuffle=False)
        predictions = model.predict(generator)
        indices, y_window = generator.get_voting_help()
        ps = []
        ys = []
        for sample in np.unique(indices):
            sample_indices = np.argwhere(indices == sample)
            sample_indices = sample_indices.reshape(len(sample_indices))
            sample_predictions = [predictions[i] for i in sample_indices]
            if 'majority' in vote_alg or 'hard' in vote_alg:
                ps.append(Evaluation.hard_vote(predictions))
            elif 'soft' in vote_alg:
                ps.append(Evaluation.soft_vote(predictions))
            ys.append(y_window[sample_indices[0]])
        return np.array(ps), np.argmax(ys, axis=1)

    @staticmethod
    def hard_vote(predictions):
        predictions = np.argmax(predictions, axis=1)
        return Counter(predictions).most_common(1)[0][0]

    @staticmethod
    def soft_vote(predictions):
        a, b = np.sum(predictions, axis=0)
        return 0 if a > b else 1  

    @staticmethod
    def evaluate(model, X, y, to_categorical, metrics, store=None, window_size=None, vote_alg='hard'):
        sep = get_separator()
        evaluation = 'Evaluation of model %s' % model

        predictions = None
        if window_size is None or window_size <= 0:
            predictions = model.predict(BatchGenerator(X, y))
        else:
            predictions, y = Evaluation.vote(model, X, y, vote_alg, window_size)

        if to_categorical or hasattr(y[0], '__len__') or len(np.array(y).shape) > 1:
            y = np.argmax(y, axis=1)
            predictions = np.argmax(predictions, axis=1)

        for metric in metrics:
            evaluation += sep
            evaluation += '%s:\n%s' % (metric.__name__, metric(y, predictions))
        
        print(evaluation)

        if store is not None:
            store_path = os.path.join(store, '%s_%s.txt' % (date.today(), os.path.splitext(os.path.basename(str(model)))[0]))
            
            if not os.path.isdir(store):
                os.makedirs(store)

            with open(store_path, 'w') as f:
                f.write(evaluation)