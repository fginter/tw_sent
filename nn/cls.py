import six
assert six.PY3, "run me with python3"

import lwvlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import keras
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Conv1D
from keras.layers.pooling import MaxPooling1D

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
from keras.layers.core import Masking, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD
from keras.callbacks import Callback,ModelCheckpoint


class Data:

    def __init__(self,wv_model):
        self.wv_model=wv_model
        self.le=LabelEncoder()

    def read(self,examples_f,args):
        classes=[]
        texts=[]
        for line in examples_f:
            line=line.strip()
            if not line:
                continue
            cls, txt=line.split("\t",1)
            classes.append(cls)
            texts.append(txt)
        cv=CountVectorizer(vocabulary=self.wv_model.w_to_dim)
        document_term_m=cv.transform(texts)

        data_matrix=np.zeros((len(texts),args.max_seq_len),dtype=np.int32)
        for row_idx,doc_row in enumerate(document_term_m):
            indices=doc_row.nonzero()[1][:args.max_seq_len]
            data_matrix[row_idx,:len(indices)]=indices
        class_indices=self.le.fit_transform(classes)
        shuffle=np.arange(len(class_indices))
        random.shuffle(shuffle)
        data_matrix=data_matrix[shuffle]
        class_indices=class_indices[shuffle].astype(np.int32)
        return class_indices, data_matrix

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sclassifier')
    parser.add_argument('--embeddings', help='embeddings file')
    parser.add_argument('--edim', type=int, default=200000, help='Embedding dimension')
    parser.add_argument('--tr-data', help='training data')
    parser.add_argument('--max-seq-len', type=int, default=30, help='training data')
    args = parser.parse_args()

    #source of embeddings
    wv_model=lwvlib.load(args.embeddings,args.edim,args.edim)

    d=Data(wv_model)
    with open(args.tr_data) as tr_f:
        class_indices, data_matrix=d.read(tr_f,args)
        class_indices_1hot=keras.utils.to_categorical(class_indices)

    
    inp_seq=Input(shape=(args.max_seq_len,), name="words", dtype='int32')
    inp_embeddings=Embedding(*wv_model.vectors.shape, input_length=args.max_seq_len, mask_zero=False, weights=[wv_model.vectors])
    inp_embeddings.trainable=False
    text_src=inp_embeddings(inp_seq)

    #gru1_out=GRU(100,name="gru1")(text_src)
    cnn1_out=Conv1D(100,2,padding="same")(text_src)

    pooled=Flatten()(MaxPooling1D(pool_size=args.max_seq_len, strides=None, padding='valid')(cnn1_out))
    do=Dropout(0.3)(pooled)
    dense1=Dense(50,activation="relu")(do)

    dense_out=Dense(np.max(class_indices)+1,activation='softmax', name="dec")(dense1)
    
    model=Model(input=[inp_seq],output=dense_out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([data_matrix],class_indices_1hot,batch_size=10,epochs=100,verbose=2,validation_split=0.2)
    
    
    

    
        
