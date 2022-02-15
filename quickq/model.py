"""Keras based deepchem DNN model.

The native deppchem DNN has been changed to pytorch, and tensorflow is
desired. Here we create a Deepchem simple dense Neural network.
"""
import os
from typing import Iterable, Union, List
import pickle
import logging

import deepchem.models
import deepchem.data
import deepchem.models.losses
import deepchem.trans
import numpy
import tensorflow.keras as ks
import tensorflow as tf


try:
    from collections.abc import Sequence as SequenceCollection
except:
    from collections import Sequence as SequenceCollection
    
class DCDNN(deepchem.models.KerasModel):
    """Adapted from deepchem RobustMultitaskRegressor.
    
    Parameters
    ----------
    n_features : int
        size of feature vector
    layer_sizes : iterable
        Neurons counds for the DNN. Length of the iterable determines the
        layer counts, and the values the number of neurons in each of those
        layers. Alternative to specifying neuron and layer count.
    neuron_count : int
        Number of neurons in each hidden layer, alternative to specifying layer_sizes
    layer_count : int
        Number of layers with neuron_count, alternative to specifting layer_sizes 
    weight_init_stdevs : iterable of float or float
        Standard deviation of random weight initialization for each or all
        layers
    bias_init_consts : iterable of float or float
        value of bias initialization for each or all layers
    weight_decay_penalty : float
        Value of weight regularization
    weight_decay_penalty_type : str
        Type of regularization eg. "l2"
    dropouts : iterable of float or float
        Dropout rates to use for each or all layers.
    activation_fns : iterable of callable or callable
        tensorflow activation functions to use for each or all layers.
    """
    def __init__(
        self,
        n_features: int,
        layer_sizes: Union[List[int], int] = None,
        neuron_count: int = None,
        layer_count: int = None,
        weight_init_stddevs: Union[List[float], float] = 0.02,
        bias_init_consts: Union[List[float], float] = 1.0,
        weight_decay_penalty: float = 0.0,
        weight_decay_penalty_type: str = "l2",
        dropouts: Union[List[float], float] = 0.0,
        activation_fns: Union[List[callable], callable] = tf.nn.relu,
        **kwargs
    ):
        if layer_sizes is not None:
            assert neuron_count is None, 'Cannot specify both layer_sizes and neuron_count.'
            assert layer_count is None, 'Cannot specify both layer_sizes and layer_count.'
        else:
            if neuron_count is None or layer_count is None:
                raise TypeError(
                    'Must specify neuron and layer count if layer_sizes not specified.'
                )
            layer_sizes = [neuron_count]*layer_count
        
        self.n_features = n_features
        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if not isinstance(activation_fns, SequenceCollection) or type(activation_fns) == str:
            activation_fns = [activation_fns] * n_layers
        if weight_decay_penalty != 0.0:
            if weight_decay_penalty_type == 'l1':
                regularizer = ks.regularizers.l1(weight_decay_penalty)
            else:
                regularizer = ks.regularizers.l2(weight_decay_penalty)
        else:
            regularizer = None
            
        def build():
        # begin with the input
            features = ks.Input(shape=(n_features,))
            prev_layer = features

            # add the DNN layers
            for size, weight_stddev, bias_const, dropout, activation_fn in zip(
                layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
                activation_fns
            ):
                if size == 0:
                     continue
                layer = ks.layers.Dense(
                    size,
                    activation=activation_fn,
                    kernel_initializer=ks.initializers.TruncatedNormal(
                        stddev=weight_stddev
                    ),
                    bias_initializer=tf.constant_initializer(value=bias_const),
                    kernel_regularizer=regularizer
                )(prev_layer)

                if dropout > 0.0:
                    layer = ks.layers.Dropout(rate=dropout)(layer)
                prev_layer = layer

            # add the output layer
            output = ks.layers.Dense(1)(prev_layer)

            model = ks.Model(inputs=features, outputs=output)
            return model
        
        model = build()
        
        # init the deepchem wrapper
        super().__init__(
            model, deepchem.models.losses.L2Loss(), output_types=['prediction'], **kwargs
        )
        return
    
    def default_generator(
        self,
        dataset: deepchem.data.Dataset,
        epochs: int = 1,
        mode: str = 'fit',
        deterministic: bool = True,
        pad_batches: bool = False
    ):
        """Default data generator for the model.
        
        Wraps the dataset iterbatches to produce data of the correct form for
        this class.
        
        Parameters
        ----------
        dataset : deepchem.data.Dataset
            dataset to iterate
        epochs : int
            Number of passes through the data
        mode : str
            ignored
        deterministic : bool, default True
            Whether to produce deterministic target values
        pad_batches : bool, default False
            Whether to pad the last batch.
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches
            ):
                yield ([X_b], [y_b], [w_b])