from keras.layers import Input, Dropout, Flatten
import keras
from keras.layers.core import Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import h5py
import cv2
import os

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

def read_data(filename, dataset):
    db = h5py.File(filename, 'r')
    data = np.array(db[dataset])
    db.close()
    return data

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess(Xtrain, Ytrain, Xtest, Ytest):
    ytrain = Ytrain.astype('int')
    ytest = Ytest.astype('int')
    _, r, c, ch = Xtrain.shape

    # Xtrain = np.array([self.gray(i) for i in Xtrain], dtype='float').reshape(-1, r, c, 1)
    # Xtest = np.array([self.gray(i) for i in Xtest], dtype='float').reshape(-1, r, c, 1)

    posytrain = ytrain[ytrain>0]
    posytest = ytest[ytest>0]
    posxtrain = Xtrain[ytrain[:,0]>0, :, :, :]
    posxtest = Xtest[ytest[:,0]>0, :, :, :]

    negytrain = ytrain[ytrain<1]
    negytest = ytest[ytest<1]
    negxtrain = Xtrain[ytrain[:,0]<1, :, :, :]
    negxtest = Xtest[ytest[:,0]<1, :, :, :]

    posytrain = posytrain[:int(len(posytrain)/2)] #2
    posytest = posytest[:int(len(posytest)/2)]
    posxtrain = posxtrain[:int(len(posxtrain)/2)]
    posxtest = posxtest[:int(len(posxtest)/2)]

    negytrain = negytrain[:int(len(negytrain)/4)] #4
    negytest = negytest[:int(len(negytest)/4)]
    negxtrain = negxtrain[:int(len(negxtrain)/4)]
    negxtest = negxtest[:int(len(negxtest)/4)]

    xtrain = np.concatenate([negxtrain, posxtrain], axis=0)
    xtest = np.concatenate([negxtest, posxtest], axis=0)
    ytrain = np.concatenate([negytrain, posytrain], axis=0)
    ytest = np.concatenate([negytest, posytest], axis=0)

    # ytrain[ytrain==0] = -1
    # ytrain[ytrain==10] = 0
    # ytrain[ytrain==-1] = 10
    # ytest[ytest==0] = -1
    # ytest[ytest==10] = 0
    # ytest[ytest==-1] = 10

    ytrain = np_utils.to_categorical(ytrain, 11)
    ytest = np_utils.to_categorical(ytest, 11)

    mean = Xtrain.mean()
    xtrain = xtrain - mean
    xtest = xtest - mean

    return xtrain, ytrain, xtest, ytest

def alt_preprocess(Xtrain, Ytrain, Xtest, Ytest):
    ytrain = Ytrain.astype('int')
    ytest = Ytest.astype('int')
    _, r, c, ch = Xtrain.shape

    # Xtrain = np.array([self.gray(i) for i in Xtrain], dtype='float').reshape(-1, r, c, 1)
    # Xtest = np.array([self.gray(i) for i in Xtest], dtype='float').reshape(-1, r, c, 1)

    xtrain = Xtrain[ytrain[:,0]>0, :, :, :]
    xtest = Xtest[ytest[:,0]>0, :, :, :]
    ytrain = ytrain[ytrain>0]
    ytrain[ytrain==10] = 0
    ytest = ytest[ytest>0]
    ytest[ytest==10] = 0

    ytrain = np_utils.to_categorical(ytrain, 10)
    ytest = np_utils.to_categorical(ytest, 10)

    # xtrain = xtrain/255
    # xtest = xtest/255

    xtrain = xtrain[:int(len(xtrain)/2)]
    ytrain = ytrain[:int(len(ytrain)/2)]
    xtest = xtest[:int(len(xtest)/2)]
    ytest = ytest[:int(len(ytest)/2)]

    mean = xtrain.mean()
    xtrain = xtrain - mean
    xtest = xtest - mean
    return xtrain, ytrain, xtest, ytest

def plot(hist):
    dir = 'report'
    loss = hist['loss']
    val_loss = hist['val_loss']
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('VGG16 Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(dir, 'vgg_acc.png'))
    plt.clf()

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('VGG16 Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(dir, 'vgg_loss.png'))
    plt.clf()

def confusion(probs, ytest):
    truth = ytest.argmax(axis=1)
    preds = probs.argmax(axis=1)
    matrix = confusion_matrix(truth, preds)
    report = classification_report(truth, preds)
    print(matrix)
    print(report)
    with open('report/rec_res.txt', 'w') as f:
        f.write(report)


def train_vgg16(pretrained=True, plot=True, save_model=True, alt=False):
    if pretrained:
        vgg = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
    else:
        vgg = keras.applications.vgg16.VGG16(weights=None, include_top=False, input_shape=(32,32,3))

    addm = Sequential()
    addm.add(Flatten(input_shape=vgg.output_shape[1:]))
    addm.add(Dense(256, activation='relu'))
    addm.add(Dense(11, activation='softmax'))

    model = Model(inputs=vgg.input, outputs=addm(vgg.output))

    Xtrain = read_data('train.hdf5', 'images')
    Ytrain = read_data('train.hdf5', 'labels')
    Xtest = read_data('test.hdf5', 'images')
    Ytest = read_data('test.hdf5', 'labels')


    xtrain, ytrain, xtest, ytest = preprocess(Xtrain, Ytrain, Xtest, Ytest)
    print('Shapes are: ', xtrain.shape, xtest.shape)
    # sgd = SGD(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(xtrain, ytrain, batch_size=200, epochs=5, validation_data=(xtest, ytest), shuffle=True, verbose=1)
    score = model.evaluate(xtest, ytest)
    print('Test Score: ', score[0])
    print('Test Accuracy: ', score[1])
    if save_model:
        model.save('models/trained_vgg16.hdf5')
    if plot:
        probs = model.predict(xtest)
        confusion(probs, ytest)
        # plot(hist.history)



if __name__ == '__main__':
    # Xtrain = read_data('train.hdf5', 'images')
    # Ytrain = read_data('train.hdf5', 'labels')
    # Xtest = read_data('test.hdf5', 'images')
    # Ytest = read_data('test.hdf5', 'labels')
    # #
    # xtrain, ytrain, xtest, ytest = alt_preprocess(Xtrain, Ytrain, Xtest, Ytest)
    # # print(xtrain.shape)
    # # print(ytrain.shape)
    # #
    # for i in range(0, 100, 4):
    #     print(ytrain[-i])
    #     cv2.imshow('sdf', xtrain[-i])
    #     cv2.waitKey(0)
    train_vgg16(pretrained=True, plot=False, save_model=False, alt=False)
