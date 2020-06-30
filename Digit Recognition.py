import numpy as np 
import pandas as pd 

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Activation, Dropout, DepthwiseConv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt

import pandas as pd

data = pd.read_csv("/content/drive/My Drive/digit-recognizer/train.csv")

test = pd.read_csv("/content/drive/My Drive/digit-recognizer/test.csv")

X = data.iloc[:,1:785]
Y = data['label']
X = X.to_numpy()
Y = Y.to_numpy()

import os
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils import to_categorical
from keras_applications.

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1,random_state = 0,stratify = Y)

X_train = np.vstack((X_train, X_test))
y_train = np.concatenate([Y_train, Y_test])
X_train = X_train.reshape(-1, 28, 28, 1)
print(X_train.shape, y_train.shape)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_train.shape

(X_tr, y_tr), (X_te, y_te) = mnist.load_data()
X_tr = np.vstack((X_tr, X_te))
y_tr = np.concatenate([y_tr, y_te])
y_val = y_tr.astype('int32')
X_val = X_tr.astype('float32')
X_val = X_val.reshape(-1,28,28,1)
y_val = to_categorical(y_val)

def create_model():
    model = Sequential()
    
    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    
    model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 192, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    
    model.add(Conv2D(filters = 192, kernel_size = 5, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2, padding = 'same'))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model = create_model()
model.summary()

reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.3, min_lr = 0.00001)
checkpoint = ModelCheckpoint('mnist_weights.h5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 10, verbose = 1, restore_best_weights = True)

callbacks = [reduce_learning_rate, checkpoint, early_stopping]

history = model.fit(X_val, 
                    y_val, 
                    batch_size = 100, 
                    epochs = 30,
                    validation_data = (X_train, y_train),  
                    callbacks = callbacks,
                    verbose = 1, 
                    shuffle = True)

test = pd.read_csv("/content/drive/My Drive/digit-recognizer/test.csv")
test = test.to_numpy()
Test = test.reshape(-1,28,28,1)

predict = model.predict(Test)
TEST = np.argmax(predict,axis=1)

from numpy import savetxt

sub = pd.read_csv('/content/drive/My Drive/digit-recognizer/sample_submission.csv')
sub['Label'] = TEST
sub.to_csv('submissio.csv',index = False)

