#importing essential libraries(keras, numpy, pandas). 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
import pandas as pd

#Inputing the data. In the third row takes care of the header
#from the input file. Creates xtrain and xtest(variables for our model)
In the third row, we get labels(numbers)
train = pd.read_csv('train.csv')
xtrain = (train.iloc[:,1:].values).astype('float32')
labels = train.iloc[:,0].values.astype('int32')
xtest = (pd.read_csv('test.csv').values).astype('float32')


#Convert vector (integers from 0) to binary class matrix, 
#for use with categorical_crossentropy
ytrain = np_utils.to_categorical(labels) 

#pre-processing: divide by max and substract mean. first we divide xtrain and xtest with their maximum,
#and then we subtract the mean value with the help of the function numpy.std
maximum = np.max(xtrain)
xtrain /= maximum
xtest /= maximum

mean = np.std(xtrain)
xtrain -= mean
xtest -= mean


#This is the actual "core" of the model. With help from keras creates neural network. 
#Activation function is rectified linear unit. 
#Dropout prevents overfitting and first argument in Dense specifies number of hidden layers.
#The shape[] function returns the dimensions of the array

input_ds = xtrain.shape[1]
num_classes = ytrain.shape[1]


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=input_ds))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



#Compiles the model. Loss function is crossentropy function, and optimizer is rms. Default optimizer would be
#SGD(Stochastic gradient descent) and default loss function 'mean_squared_error'. Rmsprop optimizer 
#is probably the best optimizer for multi-class classification problems, and therefore we stick with it.  '''
model.compile(loss='categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])


#fits the data to our model.'''
model.fit(xtrain, ytrain, nb_epoch=20, batch_size=16, validation_split=0.1, verbose=2)


#this is the actual prediction of the values.

try:
    ypred = model.predict_classes(xtest, verbose=0)
except:
    KeyboardInterrupt
    pass


#these last two lines write the outputs to csv file(creates one if not already there)
df = pd.DataFrame(ypred)
df.to_csv('submitfile.csv')






































