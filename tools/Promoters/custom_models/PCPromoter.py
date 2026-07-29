import numpy as np
from pathlib import Path
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

from typing import List

from .Meta import PromoterType, ModelType
from .Cascade import CascadeModel
    

class PCPromoter(CascadeModel):
    def __init__(self, name: str, load_paths: List[Path], order: List[PromoterType], m_type: ModelType = ModelType.SAVED_MODEL):
        super().__init__(name, load_paths, order, m_type)
    
    @staticmethod
    def _preprocess(data: List[str]) -> List[np.ndarray]:
        ## Check Seqs Length
        for i, seq in enumerate(data):   
            if ( seq_len := len(seq) ) != 81:
                raise ValueError(f"Each sequence must have a length of 81nt.\nSequence {i} has length {seq_len}nt")

        encoding = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}
        return [np.array([[encoding[x] for x in seq.upper()] for seq in data])]
    
    @staticmethod
    def _create_model():
        input_shape = (81,4)
        inputs = Input(shape = input_shape)

        convLayer = Conv1D(filters = 32, kernel_size = 7,activation = 'relu',input_shape = input_shape, kernel_regularizer = regularizers.l2(1e-5), bias_regularizer = regularizers.l2(1e-4))(inputs)
        normalizationLayer = BatchNormalization()(convLayer)
        poolingLayer = AveragePooling1D(pool_size = 2, strides=2)(normalizationLayer)
        dropoutLayer0 = Dropout(0.35)(normalizationLayer)

        convLayer2 = Conv1D(filters = 32, kernel_size = 5,activation = 'relu',kernel_regularizer = regularizers.l2(1e-4), bias_regularizer = regularizers.l2(1e-5))(dropoutLayer0)
        poolingLayer2 = MaxPooling1D(pool_size = 2, strides=2)(convLayer2)
        dropoutLayer1 = Dropout(0.30)(poolingLayer2)

        flattenLayer = Flatten()(dropoutLayer1)

        denseLayer = Dense(16, activation = 'relu',kernel_regularizer = regularizers.l2(1e-3),bias_regularizer = regularizers.l2(1e-3))(flattenLayer)
        outLayer = Dense(1, activation='sigmoid')(denseLayer)

        model = Model(inputs = inputs, outputs = outLayer)
        model.compile(loss='binary_crossentropy', optimizer= SGD(momentum = 0.95, learning_rate = 0.007), metrics=['binary_accuracy'])

        return model

    def train(self):
        return "Not implemented yet!"
