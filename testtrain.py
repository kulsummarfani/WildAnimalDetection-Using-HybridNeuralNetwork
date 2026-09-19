import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, RepeatVector
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
import pickle
from tensorflow.keras.applications import VGG19
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional, GRU

'''
path = 'aug'
X = []
Y = []
labels = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(32,32,3)
            X.append(im2arr)
            label = getLabel(name)
            Y.append(label)
            print(name+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)
print(Y.shape)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255
    

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

cnn = Sequential()
cnn.add(Conv2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn.load_weights("model/cnn_weights.hdf5")
predict = cnn.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)       

vgg = VGG19(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
for layer in vgg.layers:
    layer.trainable = False
vgg_bilstm = Sequential()
vgg_bilstm.add(vgg)
vgg_bilstm.add(Conv2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
vgg_bilstm.add(MaxPooling2D(pool_size = (1, 1)))
vgg_bilstm.add(Conv2D(32, (1, 1), activation = 'relu'))
vgg_bilstm.add(MaxPooling2D(pool_size = (1, 1)))
vgg_bilstm.add(Flatten())
vgg_bilstm.add(RepeatVector(2))
vgg_bilstm.add(Bidirectional(LSTM(32)))
vgg_bilstm.add(Dense(units = 256, activation = 'relu'))
vgg_bilstm.add(Dense(units = y_train.shape[1], activation = 'softmax'))
vgg_bilstm.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/vgg_bilstm_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/vgg_bilstm_weights.hdf5', verbose = 1, save_best_only = True)
    hist = vgg_bilstm.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/vgg_bilstm_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vgg_bilstm.load_weights("model/vgg_bilstm_weights.hdf5")

predict = vgg_bilstm.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)


extension_gru = Sequential()
extension_gru.add(Conv2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
extension_gru.add(MaxPooling2D(pool_size = (2, 2)))
extension_gru.add(Conv2D(32, (3, 3), activation = 'relu'))
extension_gru.add(MaxPooling2D(pool_size = (2, 2)))
extension_gru.add(Flatten())
extension_gru.add(RepeatVector(2))
extension_gru.add(Bidirectional(GRU(32)))
extension_gru.add(Dense(units = 256, activation = 'relu'))
extension_gru.add(Dense(units = y_train.shape[1], activation = 'softmax'))
extension_gru.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/extension_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
    hist = extension_gru.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/extension_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    extension_gru.load_weights("model/extension_weights.hdf5")
predict = extension_gru.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)       





                      
