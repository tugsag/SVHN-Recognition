import numpy as np
import cv2
import h5py
import keras
import mser
import os

DIR = 'test_input'
OUT = 'outputs'

def summaries():
    model = keras.models.load_model('models/trained_model.hdf5')
    clf = keras.models.load_model('models/trained_classifier.hdf5')
    print("Recognition Model:  ")
    print(model.summary())
    for layer in model.layers:
        print(layer.name)
        print(layer.input_shape)
    print('---------------')
    print('Classifier Model: ')
    print(clf.summary())
    for layer in clf.layers:
        print(layer.name)
        print(layer.input_shape)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def non_max(boxes, patches, probs, thresh=.3):
    boxes = np.array(boxes, dtype='float')
    pick = []

    y1 = boxes[:,0]
    y2 = boxes[:,1]
    x1 = boxes[:,2]
    x2 = boxes[:,3]

    area = (x2-x1+1) * (y2-y1+1)
    ids = np.argsort(probs)
    while len(ids)> 0:
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[ids[:last]])
        xx2 = np.minimum(x2[i], x2[ids[:last]])
        yy1 = np.maximum(y1[i], y1[ids[:last]])
        yy2 = np.minimum(y2[i], y2[ids[:last]])

        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)

        overlap = (w*h)/area[ids[:last]]
        ids = np.delete(ids, np.concatenate(([last], np.where(overlap>thresh)[0])))

    return boxes[pick].astype('int'), patches[pick], probs[pick], pick

def preprocess(patches, mean=112.5):
    ims, r, c, ch = patches.shape
    patches = np.array([gray(p) for p in patches], dtype='float')
    patches = patches - mean
    patches = patches.reshape(ims, r, c, 1)
    return patches

def alt_preprocess(patches, mean=110.5):
    patches = patches - mean
    return patches

def sift(boxes, preds):
    x1 = boxes[:,2]
    y1 = boxes[:,0]
    # print(x1, y1)
    y0 = y1[0]
    inds = [0]
    for i in range(1, len(preds)):
        if abs(y1[i] - y0) < 20:
            inds.append(i)
    return inds

def thresh_areas(boxes, patches, preds):
    boxes = boxes[preds>.9]
    patches = patches[preds>.9]
    preds = preds[preds>.9]
    return boxes, patches, preds

def thresh_vgg(boxes, patches, preds):
    boxes = boxes[preds<.1]
    patches = patches[preds<.1]
    inds = np.where(preds<.1)[0]
    preds = preds[preds<.1]
    return boxes, patches, preds, inds

def return_max(probs, l=4):
    m = probs.max(axis=1)
    # print(m)
    # m = m[m>0.5]
    # msort = sorted(m, reverse=True)
    tt = m[m>0.5].mean()
    inds = np.where(m>=tt)[0]
    return inds


def run_test(model=None, clf=None, pretrained=False):
    if not os.path.exists('graded_images'):
        os.mkdir('graded_images')
    if not os.path.exists('bad_outputs'):
        os.mkdir('bad_outputs')
    imgs = os.listdir(DIR)
    imgs = sorted(imgs, key=lambda x: int(x[0]))
    # print(imgs)
    # imgs = ['bad_2.jpg']
    for ind, i in enumerate(imgs):
        print('Processing: ', ind/len(imgs))
        im = cv2.imread(os.path.join(DIR, i))
        # im = cv2.GaussianBlur(im, (5, 5), 0)
        if ind in [0, 1, 3, 6, 8, 9]:
            im = cv2.resize(im, (225, 225))
        # if ind != 0 and ind != 2 and ind != 4 and ind != 1 and ind != 6:
        #     im = cv2.resize(im, (0,0), fx=.5, fy=.5)
        # if ind == 1:
        #     im = cv2.resize(im, (0,0), fx=.33, fy=.33)
        # if ind == 0:
        #     im = cv2.resize(im, (0,0), fx=.33, fy=.33)
        if ind == 7:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            im = cv2.resize(im, (0,0), fx=.75, fy=.75)
        if ind == 2:
            im = cv2.resize(im, (0,0), fx=.66, fy=.66)
        if ind == 4:
            im = cv2.resize(im, (0,0), fx=.66, fy=.66)
        cand_regions, cand_patches, cand_boxes = mser.region_detector(im)
        if pretrained:
            model = keras.models.load_model('models/trained_model.hdf5')
            clf = keras.models.load_model('models/trained_classifier.hdf5')

        patches = preprocess(cand_patches, mean=110.5)
        # print(patches.shape)
        clf_probs = clf.predict(patches)[:,1]
        thresh_boxes, thresh_patches, thresh_probs = thresh_areas(cand_boxes, patches, clf_probs)

        fb, fp, fprobs, pick = non_max(thresh_boxes, thresh_patches, thresh_probs)

        if len(fp) > 0:
            probs = model.predict(fp)
            if len(probs) >= 4:
                # preds = probs.argmax(axis=1)
                # print(preds)
                l = len(probs)
                inds = return_max(probs, l=l)
                preds = probs[inds].argmax(axis=1)
                fb = fb[inds]
                # if ind == 2:
                #     choo = sift(fb, preds)
                #     fb=fb[choo]
                #     preds=preds[choo]
            else:
                max_probs = probs.max(axis=1)
                #threshold
                inds = np.where(probs.max(axis=1)>.3)[0]
                max_probs = probs[inds]
                fb = fb[inds]
                # print(max_probs.max(axis=1))
                preds = max_probs.argmax(axis=1)
            # print(preds)
            # preds, ind = np.unique(preds, return_index=True)


        for j in range(len(fb)):
            y1, y2, x1, x2 = fb[j]
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(im, '{}'.format(preds[j]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if ind < 5:
            cv2.imwrite(os.path.join('graded_images', i), im)
        else:
            cv2.imwrite(os.path.join('bad_outputs', i), im)

def run_test_vgg():
    if not os.path.exists('vgg_outputs'):
        os.mkdir('vgg_outputs')
    imgs = os.listdir(DIR)
    # print(imgs)
    # imgs = ['1.jpg']
    for ind, i in enumerate(imgs):
        print('Processing: ', ind/len(imgs))
        im = cv2.imread(os.path.join(DIR, i))
        # im = cv2.GaussianBlur(im, (5, 5), 0)
        if ind in [0, 1, 3, 6, 8, 9]:
            im = cv2.resize(im, (225, 225))
        # if ind != 0 and ind != 2 and ind != 4 and ind != 1 and ind != 6:
        #     im = cv2.resize(im, (0,0), fx=.5, fy=.5)
        # if ind == 1:
        #     im = cv2.resize(im, (0,0), fx=.33, fy=.33)
        # if ind == 0:
        #     im = cv2.resize(im, (0,0), fx=.33, fy=.33)
        if ind == 7:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            im = cv2.resize(im, (0,0), fx=.75, fy=.75)
        if ind == 2:
            im = cv2.resize(im, (0,0), fx=.66, fy=.66)
        if ind == 4:
            im = cv2.resize(im, (0,0), fx=.66, fy=.66)
        cand_regions, cand_patches, cand_boxes = mser.region_detector(im)
        model = keras.models.load_model('models/trained_vgg16.hdf5')

        patches = alt_preprocess(cand_patches, mean=110.5)
        probs = model.predict(patches)
        thresh_boxes, thresh_patches, thresh_probs, inds = thresh_vgg(cand_boxes, patches, probs[:,0])
        # print(thresh_probs)
        probs = probs[inds]
        fb, fp, fprobs, picks = non_max(thresh_boxes, thresh_patches, thresh_probs)
        # print(fprobs)
        # print(probs[picks])
        preds = np.argmax(probs[picks], axis=1)


        for j in range(len(fb)):
            y1, y2, x1, x2 = fb[j]
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 1)
            if preds[j] == 10:
                preds[j] = 0
            cv2.putText(im, '{}'.format(preds[j]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imwrite(os.path.join('vgg_outputs',i), im)





if __name__ == '__main__':
    # print(sorted(os.listdir('test_input'), key=lambda x: int(x[0])))
    run_test(pretrained=True)
    # run_test_vgg()
