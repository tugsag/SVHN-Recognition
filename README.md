Classification of digits using SVHN dataset using CNN

To get grading images using trained models, execute run.py from the command line like the following:

```
python run.py -c
```
The grading images will be found in the ```graded_images``` directory. Please ensure you have the ```test_input``` folder containing the grading images. By default, this will also create the bad examples in a separate directory.

To train from scratch using SVHN training dataset, execute run.py as the following (about 2.2 hour runtime):

```
python run.py -f [training-images-directory]
```

Training images directory is the train.tar.gz dataset from http://ufldl.stanford.edu/housenumbers/. By default, the code looks for ```train``` directory in the current directory.

Execute run.py with ``` -h ``` option for more options for testing.

Requirements:
1. numpy==1.19.5
2. keras
3. tensorflow
4. h5py
5. opencv-python==4.1.2
6. matplotlib==3.1.1
7. scikit-learn
