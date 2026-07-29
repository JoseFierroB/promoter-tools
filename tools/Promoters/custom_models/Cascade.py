from abc import abstractmethod
from typing import List
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model

from .Meta import MetaModel, PromoterType, ModelType

class CascadeModel(MetaModel):

    def __init__(self, name: str, load_paths: List[Path], order: List[PromoterType], m_type: ModelType = ModelType.SAVED_MODEL, multiple_outs = False):
        self.name = name
        self.order = order
        self.models = []
        self.multiple_outs = multiple_outs
        if m_type == ModelType.SAVED_MODEL:
            for path in load_paths:
                self.models.append(load_model(path))
        else:
            for path in load_paths:
                self.models.append(self._create_model())
                self.models[-1].load_weights(path)
    
    @staticmethod
    @abstractmethod
    def _preprocess(data) -> List[np.ndarray]:
        pass
    
    @staticmethod
    @abstractmethod
    def _create_model():
        pass
    
    def predict(self, data: List[str]):
        # So much memory usage!!!
        tmp_preds = {seq : PromoterType.NON_PROMOTER for seq in data}
        preds = np.full(len(data), PromoterType.NON_PROMOTER, dtype=object)

        data_array = np.array(data)
        encoded = self._preprocess(data)

        if self.multiple_outs:
            predictions = self.models[0].predict(encoded).argmax(axis=1).ravel()
        else:
            predictions = self.models[0].predict(encoded).round().ravel()

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
        
        index = 0 # Promoter type from order list
        while still_left and index < len(self.models) - 1:
            # Iterate over all promoter models, i.e models[1:]
            model = self.models[index + 1]

            if self.multiple_outs:
                predictions = model.predict(encoded).argmax(axis=1).ravel()
            else:
                predictions = model.predict(encoded).round().ravel()

            # Get index for future use
            indices_zero, indices_nonzero = self._generate_indices(predictions)

            # Update return_preds
            tmp_preds.update({seq : self.order[index] for seq in data_array[indices_nonzero]})

            if model is not self.models[-1]:
                # Update arrays for next stage
                data_array = data_array[indices_zero]
                for i in range(len(encoded)):
                    encoded[i] = encoded[i][indices_zero]
            else:
                # Update tmp_preds with last promoter, because it has no model
                tmp_preds.update({seq : self.order[-1] for seq in data_array[indices_zero]})

            if data_array.size == 0:
                # Nothing left to classify
                still_left = False
            
            index += 1
                
        
        for i, seq in enumerate(data):
            preds[i] = tmp_preds[seq]

        return preds
    
    @abstractmethod
    def train(self):
        pass
