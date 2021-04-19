import cv2
import json
import numpy as np
import h5py
import keras
from keras.utils import np_utils

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_rec(Xtest, Ytest):
    ytest = Ytest.astype('int')
    _, r, c, ch = Xtest.shape

    Xtest = np.array([gray(i) for i in Xtest], dtype='float').reshape(-1, r, c, 1)

    xtest = Xtest[ytest[:,0]>0, :, :, :]
    ytest = ytest[ytest>0]
    ytest[ytest==10] = 0
    ytest = np_utils.to_categorical(ytest, 10)

    xtest = xtest - 112.5

    return xtest, ytest

def preprocess_clf(Xtest, Ytest):
    ytest = Ytest.astype('int')
    _, r, c, ch = Xtest.shape
    Xtest = np.array([gray(i) for i in Xtest], dtype='float').reshape(-1, r, c, 1)
    ytest[ytest>0] = 1
    ytest = np_utils.to_categorical(ytest, 2)
    xtest = Xtest - 110.5

    return xtest, ytest

def read_data(filename, dataset):
    db = h5py.File(filename, 'r')
    data = np.array(db[dataset])
    db.close()
    return data

def test_val():
    Xtest = read_data('test.hdf5', 'images')
    Ytest = read_data('test.hdf5', 'labels')

    xtest_clf, ytest_clf = preprocess_clf(Xtest, Ytest)
    xtest_rec, ytest_rec = preprocess_rec(Xtest, Ytest)

    model = keras.models.load_model('models/trained_model.hdf5')
    clf = keras.models.load_model('models/trained_classifier.hdf5')

    score1 = model.evaluate(xtest_rec, ytest_rec)
    print('Recognizer Model Scoring')
    print('Test Score: ', score1[0])
    print('Test Accuracy: ', score1[1])

    score2 = clf.evaluate(xtest_clf, ytest_clf)
    print('Classifier Model Scoring')
    print('Test Score: ', score2[0])
    print('Test Accuracy: ', score2[1])
