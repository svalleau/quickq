"""Wrapper function for tranformation and prediction."""
import pickle
import os

import quickq.predictors

filepath = os.path.dirname(os.path.realpath(__file__))
transformers_dir = os.path.join(filepath, 'transformers')

fileobj = open(f'{transformers_dir}/devtX0.pkl', 'rb')
qest_X_trans = pickle.load(fileobj)
fileobj.close()

fileobj = open(f'{transformers_dir}/devty0.pkl', 'rb')
qest_y_trans = pickle.load(fileobj)
fileobj.close()

def predict_qest(dataset):
    """Make predictions on a Qest format dataset.
    
    Transforms X, makes predictions, and untransforms y.
    
    Parameters
    ----------
    dataset : deechem.Dataset
        data to make preidctions on. Should be created by data loader.
    """
    # make transformation
    dataset_ = dataset.transform(qest_X_trans)
    y_hat = quickq.predictors.Qest.predict(dataset_)
    y_hat = qest_y_trans.untransform(y_hat)
    return y_hat

def predict_qests(dataset):
    """Make predictions on a QesTS format dataset.
    
    Parameters
    ----------
    dataset : deechem.Dataset
        data to make preidctions on. Should be created by data loader.
    """
    y_hat = quickq.predictors.QesTS.predict(dataset)
    return y_hat
    