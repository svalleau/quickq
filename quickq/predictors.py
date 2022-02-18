"""Loads trained models from hyperparameters and trained params."""
import numpy as np
import os
import json
import quickq.model

filepath = os.path.dirname(os.path.realpath(__file__))
trained_models_dir = os.path.join(filepath, 'model_params')

file = open(trained_models_dir+'/hyperopt_best_params.json', 'r')
hyper_params = json.load(file)
file.close()
file = open(trained_models_dir+'/hyperopt_rxn_best_params.json', 'r')
hyper_params_rxn = json.load(file)
file.close()


Qest = quickq.model.DCDNN(901, **hyper_params)
Qest.restore(model_dir=trained_models_dir+'/trained_qest')

QesTS = quickq.model.DCDNN(903, **hyper_params_rxn)
QesTS.restore(model_dir=trained_models_dir+'/trained_qests')