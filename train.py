from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.regularizers import l2
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


class Model:
    def __init__(self):
        self.model = self.build_model()

    # def build_model(self):
    #     model = Sequential()
    #
    #     model.add(Conv2D(32, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(2, 2))
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(64, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(128, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(2, 2))
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(180, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(256, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(2, 2))
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(280, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(300, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(2, 2))
    #     model.add(Dropout(.3))
    #
    #     model.add(Conv2D(320, kernel_size=5, kernel_initializer='he_uniform', kernel_regularizer=l2(.0005), padding='same', input_shape=(32,32,3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(.3))
    #
    #     #Dense
    #     model.add(Flatten())
    #     model.add(Dense(2800, kernel_regularizer=l2(.0005)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(.3))
    #
    #     model.add(Dense(1200, kernel_regularizer=l2(.0005)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(.3))
    #
    #     model.add(Dense(10, kernel_regularizer=l2(.0005), activation='softmax'))
    #
    #     return model

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=5, padding='same', input_shape=(32, 32, 1)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, kernel_size=5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64, kernel_size=5))
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(.3))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(.3))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model

    def gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def preprocess(self, Xtrain, Ytrain, Xtest, Ytest):
        ytrain = Ytrain.astype('int')
        ytest = Ytest.astype('int')
        _, r, c, ch = Xtrain.shape

        Xtrain = np.array([self.gray(i) for i in Xtrain], dtype='float').reshape(-1, r, c, 1)
        Xtest = np.array([self.gray(i) for i in Xtest], dtype='float').reshape(-1, r, c, 1)

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

        mean = xtrain.mean()
        xtrain = xtrain - mean
        xtest = xtest - mean

        # train_gen = ImageDataGenerator(rotation_range=15, zoom_range=.1, width_shift_range=.3, height_shift_range=.3)
        # train_gen.fit(xtrain)
        # test_set = train_gen.flow(xtest, ytest, batch_size=256)
        # return train_gen, test_set, xtrain, ytrain, xtest, ytest


        return xtrain, ytrain, xtest, ytest, mean

    def train(self, Xtrain, Ytrain, Xtest, Ytest, batch_size=200, epochs=10, save_model=False, plot=True):
        xtrain, ytrain, xtest, ytest, mean = self.preprocess(Xtrain, Ytrain, Xtest, Ytest)
        print('Shapes are: ', xtrain.shape, xtest.shape, mean)
        # sgd = SGD(learning_rate=0.1)
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))
        hist = self.model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest), shuffle=True)
        score = self.model.evaluate(xtest, ytest)
        print('Test Score: ', score[0])
        print('Test Accuracy: ', score[1])
        if save_model:
            self.model.save('models/trained_model.hdf5')
        if plot:
            probs = self.model.predict(xtest)
            self.confusion(probs, ytest)
            self.plot(hist.history)
        return self.model

    def confusion(self, probs, ytest):
        truth = ytest.argmax(axis=1)
        preds = probs.argmax(axis=1)
        matrix = confusion_matrix(truth, preds)
        report = classification_report(truth, preds)
        print(matrix)
        print(report)
        # with open('report/rec_res.txt', 'w') as f:
        #     f.write(report)

    def plot(self, hist):
        dir = 'report'
        loss = hist['loss']
        val_loss = hist['val_loss']
        acc = hist['accuracy']
        val_acc = hist['val_accuracy']
        epochs = range(1, len(loss) + 1)

        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('Recognizer Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(os.path.join(dir, 'rec_acc.png'))
        plt.clf()

        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Recognizer Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(os.path.join(dir, 'rec_loss.png'))
        plt.clf()


class Classifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=5, padding='same', input_shape=(32, 32, 1)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, kernel_size=5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64, kernel_size=5))
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(.3))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    def gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def preprocess(self, Xtrain, Ytrain, Xtest, Ytest):
        ytrain = Ytrain.astype('int')
        ytest = Ytest.astype('int')
        _, r, c, ch = Xtrain.shape

        Xtrain = np.array([self.gray(i) for i in Xtrain], dtype='float').reshape(-1, r, c, 1)
        Xtest = np.array([self.gray(i) for i in Xtest], dtype='float').reshape(-1, r, c, 1)

        ytrain[ytrain>0] = 1
        ytest[ytest>0] = 1

        ytrain = np_utils.to_categorical(ytrain, 2)
        ytest = np_utils.to_categorical(ytest, 2)

        mean = Xtrain.mean()
        xtrain = Xtrain - mean
        xtest = Xtest - mean

        return xtrain, ytrain, xtest, ytest, mean

    def train(self, Xtrain, Ytrain, Xtest, Ytest, batch_size=200, epochs=5, save_model=False, plot=True):
        xtrain, ytrain, xtest, ytest, mean = self.preprocess(Xtrain, Ytrain, Xtest, Ytest)
        print('Shapes are: ', xtrain.shape, xtest.shape, mean)
        # xtrain = xtrain[int(len(xtrain)/3):]
        # ytrain = ytrain[int(len(ytrain)/3):]
        # xtest = xtest[int(len(xtest)/3):]
        # ytest = ytest[int(len(ytest)/3):]
        # for i in range(0, 100, 4):
        #     print(ytest[-i])
        #     cv2.imshow('sdf', xtest[-i])
        #     cv2.waitKey(0)
        # adam = Adam(learning_rate=.001)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = self.model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest), shuffle=True)
        score = self.model.evaluate(xtest, ytest)
        print('Test Score: ', score[0])
        print('Test Accuracy: ', score[1])
        if save_model:
            self.model.save('models/trained_classifier.hdf5')
        if plot:
            probs = self.model.predict(xtest)
            self.confusion(probs, ytest)
            self.plot(hist.history)
        return self.model

    def confusion(self, probs, ytest):
        truth = ytest.argmax(axis=1)
        preds = probs.argmax(axis=1)
        matrix = confusion_matrix(truth, preds)
        report = classification_report(truth, preds, target_names=['None', 'Number'])
        print(matrix)
        print(report)
        # with open('report/clf_res.txt', 'w') as f:
        #     f.write(report)

    def plot(self, hist):
        dir = 'report'
        loss = hist['loss']
        val_loss = hist['val_loss']
        acc = hist['accuracy']
        val_acc = hist['val_accuracy']
        epochs = range(1, len(loss) + 1)

        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('Classifier Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(os.path.join(dir, 'clf_acc.png'))
        plt.clf()

        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Classifier Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(os.path.join(dir, 'clf_loss.png'))
        plt.clf()

def read_data(filename, dataset):
    db = h5py.File(filename, 'r')
    data = np.array(db[dataset])
    db.close()
    return data

def run_all(save=True, plot=True):
    if not os.path.exists('report'):
        os.mkdir('report')
    Xtrain = read_data('train.hdf5', 'images')
    Ytrain = read_data('train.hdf5', 'labels')
    Xtest = read_data('test.hdf5', 'images')
    Ytest = read_data('test.hdf5', 'labels')
    # for i in range(0, 100, 4):
    #     print(Ytest[i])
    #     cv2.imshow('sdf', Xtest[i])
    #     cv2.waitKey(0)
    # # #

    m = Model()
    Mm = m.train(Xtrain, Ytrain, Xtest, Ytest, save_model=save, plot=plot)
    c = Classifier()
    Cm = c.train(Xtrain, Ytrain, Xtest, Ytest, save_model=save, plot=plot)

    return Mm, Cm

if __name__ == '__main__':

    run_all(save=False, plot=False)
