import numpy as np
from pathlib import Path
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, Input, Concatenate
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model

# For _preprocess
import pandas as pd
from itertools import product
from sklearn.preprocessing import scale
from tensorflow.keras.utils import to_categorical

from typing import List

from .Meta import PromoterType, ModelType
from .Cascade import CascadeModel
    

class IPromoter(CascadeModel):
    def __init__(self, name: str, load_paths: List[Path], order: List[PromoterType], m_type: ModelType = ModelType.SAVED_MODEL):
        super().__init__(name, load_paths, order, m_type, True)
    
    @staticmethod
    def _preprocess(data: List[str]) -> List[np.ndarray]:
        ## Check Seqs Length
        for i in range(len(data)):  
            data[i] = data[i].upper()
            if ( seq_len := len(data[i]) ) != 81:
                raise ValueError(f"Each sequence must have a length of 81nt.\nSequence {i} has length {seq_len}nt")        

        ## Feature Extraction ##

        # Structural Properties of Di Nucleotide
        di_prop = pd.read_csv(Path(__file__).parent / 'data/DNA_Di_Prop.txt')
        di_prop = di_prop.iloc[:, 1:]
        scaled_di_prop = scale(di_prop, axis=1) # Standardization
        di_cols = di_prop.columns.tolist()
        di_prop = pd.DataFrame(scaled_di_prop, columns=di_cols)
        pp_di = {}
        for i in range(16):
            key = di_prop.columns[i]
            items = di_prop.iloc[:, i].tolist()
            pp_di[key] = items
        # Structural Properties of tri nucleotide
        tri_prop = pd.read_csv(Path(__file__).parent / 'data/DNA_Tri_Prop.txt')
        tri_prop = tri_prop.iloc[:, 1:]
        scaled_tri_prop = scale(tri_prop, axis=1) # Standardization
        tri_cols = tri_prop.columns.tolist()
        tri_prop = pd.DataFrame(scaled_tri_prop, columns=tri_cols)
        pp_tri = {}
        for i in range(64):
            key = tri_prop.columns[i]
            items = tri_prop.iloc[:, i].tolist()
            pp_tri[key] = items
        # Mono-mer Feature
        X1 = np.empty((len(data), len(data[0])))
        alphabet = 'ATGC'
        for i in range(len(data)):
            for j in range(len(data[0])):
                X1[i, j] = next((k for k, letter in enumerate(alphabet) if letter == data[i][j]))

        X1 = to_categorical(X1, num_classes=len(alphabet))
        # Tri-mer Feature
        lookup_table = []
        for p in product(alphabet, repeat=3):
            w = ''.join(p)
            lookup_table.append(w)
        X2 = np.empty((len(data), len(data[0])-2))
        for i in range(len(data)):
            for j in range(len(data[0])-2):
                w = data[i][j:j+3]
                X2[i,j] = lookup_table.index(w)      
        X2 = to_categorical(X2, len(lookup_table))
        # Di nucleotide properties feature
        X3 = np.empty([len(data), 80, 90], dtype=float)
        for i in range(len(data)):
            for j in range(80):
                word = data[i][j:j+2]
                value = pp_di[word]
                for k in range(90):
                    X3[i, j, k] = value[k]
        # Tri nucleotide properties feature
        X4 = np.empty([len(data), 79, 12], dtype=float)
        for i in range(len(data)):
            for j in range(79):
                word = data[i][j:j+3]
                value = pp_tri[word]
                for k in range(12):
                    X4[i, j, k] = value[k]
        return [X1, X2, X3, X4]

    
    @staticmethod
    def _create_model():
        initializer = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Mono-Mer
        inputs_monomer = Input(shape=(81,4,))
        x = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs_monomer) # , name='Mono-Mer_Conv1'
        x = Dropout(0.5)(x)
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(x) # 50
        x = Dropout(0.5)(x)
        x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        monomer_last = Flatten()(x)

        # Tri-Mer
        inputs_trimer = Input(shape=(79,64,))
        x = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs_trimer) # , name='Tri-Mer_Conv1'
        x = Dropout(0.5)(x)
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(x) # 40
        x = Dropout(0.5)(x)
        x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        trimer_last = Flatten()(x)

        # Di-nucleotide Structural Prop branch
        inputs_dinuc = Input(shape=(80,90,))
        x = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs_dinuc) # , name='Di-nuc_Conv1'
        x = Dropout(0.5)(x)
        x = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        dinuc_last = Flatten()(x)

        # Tri-nucleotide Structural Prop branch
        inputs_trinuc = Input(shape=(79,12,))
        x = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs_trinuc) # , name='Tri-nuc_Conv1'
        x = Dropout(0.5)(x)
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(x)
        x = Dropout(0.5)(x)
        trinuc_last = Flatten()(x)

        # Concatenation
        x = Concatenate()([monomer_last, trimer_last, dinuc_last, trinuc_last])
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)


        model = Model(inputs=[inputs_monomer, inputs_trimer, inputs_dinuc, inputs_trinuc], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self):
        return "Not implemented yet!"
