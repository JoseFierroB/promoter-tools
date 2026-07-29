from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from enum import Enum, auto

# Since PromoterType's values are accessed sometimes, auto isn't used 
class PromoterType(Enum):
    NON_PROMOTER = 1
    SIGMA_70     = 2
    SIGMA_24     = 3
    SIGMA_28     = 4
    SIGMA_38     = 5
    SIGMA_32     = 6
    SIGMA_54     = 7
    
class ModelType(Enum):
    SAVED_MODEL = auto()
    WEIGHTS_ONLY = auto()

class CustomMetrics:
    """
    Todo: Unify methods
    """     
    @staticmethod
    def compute_TnF(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tp, tn, fp, fn
    
    # Training methods
    @staticmethod
    def matthews_correlation_coefficient(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return num / K.sqrt(den + K.epsilon())
   
    @staticmethod
    def specificity(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())
    
    @staticmethod
    def sensitivity(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tp / (tp + fn + K.epsilon())
    
    # Evaluation methods
    @staticmethod
    def val_matthews_correlation_coefficient(*, tp, fp, tn, fn):
        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return num / K.sqrt(den + K.epsilon())
   
    @staticmethod
    def val_specificity(*, tp, fp, tn, fn):
        return tn / (tn + fp + K.epsilon())
    
    @staticmethod
    def val_sensitivity(*, tp, fp, tn, fn):
        return tp / (tp + fn + K.epsilon())
    
    @staticmethod
    def val_accuracy(*, tp, fp, tn, fn):
        return (tp + tn) / (tp + tn + fp + fn + K.epsilon())

class MetaModel(ABC):

    @abstractmethod
    def __init__(self, name: str):
        self.name = name
    
    @staticmethod
    @abstractmethod
    def _preprocess(data):
        pass
    
    # Maybe not
    @staticmethod
    @abstractmethod
    def _create_model():
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @staticmethod
    def _generate_indices(predictions: np.ndarray, comp_value = 0):
        # Get index for future use
        bindices_zero = (predictions == comp_value) # Boolean indices
        indices_zero = np.arange(len(predictions))[bindices_zero] # For next stage
        indices_nonzero = np.arange(len(predictions))[~bindices_zero] # Classification
        return indices_zero, indices_nonzero
    
    def test(self, inputs: List[str], outputs: List[PromoterType]):
        # Create a list holding all possible promoter types (+ non promoter)
        types = list(PromoterType)
        # Get predictions for later use
        preds_array = self.predict(inputs)
        # Create np.array of outputs for later use
        outputs_array = np.array(outputs)
        # Get indices for each promoter type from outputs_array, store them in a dictionary
        outputs_idx = {_type : self._generate_indices(outputs_array, _type) for _type in types}
        
        stats = {}
        # Calculate metrics for each type
        for _type, idx in outputs_idx.items():
            tp = tf.constant(np.count_nonzero(preds_array[idx[0]] == _type), dtype=tf.float32)
            fp = tf.constant(np.count_nonzero(preds_array[idx[1]] == _type), dtype=tf.float32)
            fn = tf.constant(np.count_nonzero(preds_array[idx[0]] != _type), dtype=tf.float32)
            tn = tf.constant(np.count_nonzero(preds_array[idx[1]] != _type), dtype=tf.float32)
            #print(f"Type: {_type} --- TP: {tp.numpy()} --- FP: {fp.numpy()} --- TN: {tn.numpy()} --- FN: {fn.numpy()}")
            # Metrics
            sn = CustomMetrics.val_sensitivity(tp=tp, fp=fp, fn=fn, tn=tn).numpy()
            sp = CustomMetrics.val_specificity(tp=tp, fp=fp, fn=fn, tn=tn).numpy()
            cc = CustomMetrics.val_matthews_correlation_coefficient(tp=tp, fp=fp, fn=fn, tn=tn).numpy()
            ac = CustomMetrics.val_accuracy(tp=tp, fp=fp, fn=fn, tn=tn).numpy()
            #print(f"Sn: {sn} --- Sp: {sp} --- MCC: {cc} --- Ac: {ac}")
            
            stats[_type] = {"TP": tp.numpy(), "FP": fp.numpy(), 
                            "TN": tn.numpy(), "FN": fn.numpy(), 
                            "Specificity": sp, "Sensitivity": sn, 
                            "MCC": cc, "Accuracy": ac
                           }

        total_acc = sum(1 for x, y in zip(preds_array, outputs) if x == y) / len(inputs)
        return stats, total_acc