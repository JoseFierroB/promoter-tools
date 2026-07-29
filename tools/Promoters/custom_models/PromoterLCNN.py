import numpy as np

from pathlib import Path
from typing import List

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization
from tensorflow.keras.layers import MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.losses import Reduction
from tensorflow.keras import Model, regularizers

from tensorflow_addons.optimizers import LazyAdam
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from .Meta import MetaModel, ModelType, PromoterType

class PromoterLCNN(MetaModel):

    def __init__(self, name: str, load_paths: List[Path], m_type: ModelType = ModelType.SAVED_MODEL):
        self.name = name
        self.models = []
        if (p_len := len(load_paths)) != 2:
            raise ValueError(f"'load_paths' must be a list of two items, got {p_len} instead")
        if m_type == ModelType.SAVED_MODEL:
            for path in load_paths:
                self.models.append(load_model(path))
        else:
            for path in load_paths:
                is_first = len(self.models) == 0
                self.models.append(self._create_model(is_first))
                self.models[-1].load_weights(path)
    
    @staticmethod
    def _preprocess(data: List[str]) -> List[np.ndarray]:
        ## Check Seqs Length
        for i, seq in enumerate(data):   
            if ( seq_len := len(seq) ) != 81:
                raise ValueError(f"Each sequence must have a length of 81nt.\nSequence {i} has length {seq_len}nt")

        encoding = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}
        return [np.array([[encoding[x] for x in seq.upper()] for seq in data])]
    
    @staticmethod
    def _create_model(types: bool):
        if types:
            out_classes = 6
        else:
            out_classes = 2
        # input
        input_ = Input(shape =(81,4))

        # 1st Conv Block
        # Params for first Conv1D
        hp_filters_1 = 128
        hp_kernel_1 = 5
        hp_kreg_1 = 1e-3
        hp_breg_1 = 1e-2
        # Params for second Conv1D
        hp_filters_2 = 128
        hp_kernel_2 = 9
        hp_kreg_2 = 1e-3
        hp_breg_2 = 1e-5
        # Params for Dropout
        hp_drop_1 = 0.45

        x = Conv1D (filters=hp_filters_1, kernel_size=hp_kernel_1, padding ='same', activation='relu', kernel_regularizer = regularizers.l2(hp_kreg_1), bias_regularizer = regularizers.l2(hp_breg_1))(input_)
        x = Conv1D (filters=hp_filters_2, kernel_size=hp_kernel_2, padding ='same', activation='relu', kernel_regularizer = regularizers.l2(hp_kreg_2), bias_regularizer = regularizers.l2(hp_breg_2))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size =2, strides =2, padding ='same')(x)
        x = Dropout(rate=hp_drop_1)(x)

        # Fully connected layers
        x = Flatten()(x)
        hp_units = 32
        x = Dense(units=hp_units, activation ='relu', kernel_regularizer = regularizers.l2(1e-3), bias_regularizer = regularizers.l2(1e-3))(x)
        output = Dense(units = out_classes, activation ='softmax')(x)

        # Creating the model
        model = Model (inputs=input_, outputs=output)
        model.compile(optimizer=LazyAdam(), loss=SigmoidFocalCrossEntropy(alpha=0.20, gamma=3, reduction=Reduction.AUTO), metrics=['accuracy'])

        return model
    
    def predict(self, data: List[str]):
        # So much memory usage!!!
        tmp_preds = {seq : PromoterType.NON_PROMOTER for seq in data}
        preds = np.full(len(data), PromoterType.NON_PROMOTER, dtype=object)

        data_array = np.array(data)
        encoded = self._preprocess(data)

        predictions = self.models[0].predict(encoded).argmax(axis=1).ravel()

        # Get index for future use
        indices_zero, indices_nonzero = self._generate_indices(predictions)

        # Do not update return preds, as NON_PROMOTER is the default value

        # Update arrays for next stage
        # NOTE: Since Promoter is the positive class, and only
        # in this first prediction, the nonzero indices are
        # passed to the next model
        data_array = data_array[indices_nonzero]
        for i in range(len(encoded)):
            encoded[i] = encoded[i][indices_nonzero]
        
        still_left = True

        if data_array.size == 0:
            # Nothing left to classify
            still_left = False
        
        if still_left:
            # Second stage, multiclass model
            predictions = self.models[1].predict(encoded).argmax(axis=1).ravel()
            # Update return_preds
            tmp_preds.update({seq : PromoterType(predictions[i] + 2) for i, seq in enumerate(data_array)})
            # NOTE: That +2 offset appears as a consequence of the training
        
        for i, seq in enumerate(data):
            preds[i] = tmp_preds[seq]

        return preds

    def train(self):
        return "Not implemented yet!"