import tensorflow as tf

import keras
from keras.src.models.sequential import Sequential
from keras import layers as klayers
from keras import optimizers as koptimizers
import numpy as np
from scipy.optimize import minimize

from typing import List, Callable

class trainer_model(Sequential):
    def __init__(self,
                 layers:klayers.Layer=None,
                 trainable:bool=True,
                 name:str=None,
                 ):
        super().__init__(layers=layers, trainable=trainable, name=name)

    # TODO: gradient based methods
    def back_minimize(self,
                 x0:np.ndarray=None,
                 method = 'BFGS', verbose = 0):
        """
        After the model is trained, minimize the output by training the input.
        """

        # # @tf.function
        def to_minimize(x):
            pad_x = np.array([x])
            return self(pad_x)

        if x0 is None:
            x = np.random.rand(self.inputs[0].shape[1])
        else:
            x = x0
        
        result = minimize(to_minimize, x, method=method, tol=1e-6)

        return result.x

    @staticmethod
    def default_model(input_shape:tuple):
        initializer = keras.initializers.he_normal(seed=10)
        return trainer_model(
            layers=[
                klayers.Input(input_shape),
                klayers.Dense(96, activation='elu',
                                   kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(1e-8),
                                   ),
                klayers.Dense(64, activation='elu'),
                klayers.Dense(18, activation='elu'),
                klayers.Dense(10, activation='elu'),
                klayers.Dense(1),
                ],
                name='default_model'
            )

    @staticmethod
    def simple_model(input_shape:tuple):
        return trainer_model(
            layers=[
                klayers.Input(input_shape),
                klayers.Dense(32, activation='elu'),
                klayers.Dense(8, activation='sigmoid'),
                klayers.Dense(1),
                ],
                name='simple_model'
            )