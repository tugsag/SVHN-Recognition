import h5py
import dataset
import train
import detect
import test
import trainvgg
import sys
import os

if __name__ == '__main__':
    arglist = sys.argv[1:]
    if len(arglist) < 1:
        print('''
            -c: to run trained models on grading images
            -f ['train-imgs-dir']: full runthrough, default option is 'train' directory

            For testing and quick checks, use the following:
            -t: to train from precompiled datasets,
            -m: to run trained models on validation sets,
            -v: to run trained VGG16 model on grading images (saves to different outputs directory),
            -s: summary of models

            Please provide the training images from SVHN found on README,
            precompiled datasets and trained models are provided by default''')
    else:
        if arglist[0] == '-h':
            print('''
                -c: to run trained models on grading images,
                -f ['train-imgs-dir']: full runthrough, default option is 'train' directory (about 2.2 hours runtime)

                For testing and quick checks, use the following:
                -n ['train-imgs-dir']: to run everything without training VGG16 network (about 55 minute runtime)
                -t: to train main models from precompiled datasets (about 2 hours runtime),
                -m: to run trained models on validation sets,
                -v: to run trained VGG16 model on grading images (saves to different outputs directory),
                -s: summary of models

                Please provide the training images from SVHN found on README,
                precompiled datasets and trained models are provided by default''')

        if arglist[0] == '-f':
            if not os.path.exists('models'):
                os.mkdir('models')
            dir = 'train'
            if len(arglist) > 1:
                dir = arglist[1]
            print("Starting dataset preprocessing from ", dir)
            dataset.compile(DIR=dir)
            print("Training")
            train.run_all(save=True, plot=True)
            trainvgg.train_vgg16()
            print("Starting detection")
            detect.run_test(pretrained=True)
            detect.run_test_vgg()

        if arglist[0] == '-n':
            dir = 'train'
            if len(arglist) > 1:
                dir = arglist[1]
            print(dir)
            dataset.compile(DIR=dir)
            train.run_all(save=True)
            detect.run_test(pretrained=True)

        elif arglist[0] == '-t':
            print("Training")
            train.run_all(save=True)
            trainvgg.train_vgg16()
            print("Starting detection")
            detect.run_test(pretrained=True)
            detect.run_test_vgg()

        elif arglist[0] == '-m':
            test.test_val()

        elif arglist[0] == '-c':
            detect.run_test(pretrained=True)
            detect.run_test_vgg()

        elif arglist[0] == '-s':
            detect.summaries()

        elif arglist[0] == '-v':
            detect.run_test_vgg()

        else:
            print('''
                -c: to run trained models on grading images,
                -f ['train-imgs-dir']: full runthrough, default option is 'train' directory

                For testing and quick checks, use the following:
                -t: to train from precompiled datasets,
                -m: to run trained models on validation sets,
                -v: to run trained VGG16 model on grading images (saves to different outputs directory),
                -s: summary of models

                Please provide the training images from SVHN found on README,
                precompiled datasets and trained models are provided by default''')
