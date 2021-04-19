import numpy as np
import mser
import os
import json
import cv2
import sys
import h5py
np.random.seed(42)

positive_samples = []
positive_labels = []
negative_samples = []


### Load Datasets ###
### Helpers ###

def load_ann():
    return json.loads(open('train_ann.json').read())

def get_truth(cont):
    boxes = []
    labels = []
    for b in cont['boxes']:
        x1 = int(b['left'])
        y1 = int(b['top'])
        w = int(b['width'])
        h = int(b['height'])

        boxes.append((y1, y1+h, x1, x1+w))
        labels.append(int(b['label']))

    return np.array(boxes), np.array(labels)

def get_patches(img, boxes):
    h, w = img.shape[0], img.shape[1]
    patches = []
    for box in boxes:
        y1,y2,x1,x2 = box
        (x1, y1) = (max(x1, 0), max(y1, 0))
        (x2, y2) = (min(x2, w), min(y2, h))
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            patches.append(np.ones((32, 32, 3)))
            continue
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patches.append(patch)

    return np.array(patches)

def calc_overlaps(boxes, true_boxes):
    overlaps = []
    for tb in true_boxes:
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y1gt = tb[0]
        y2gt = tb[1]
        x1gt = tb[2]
        x2gt = tb[3]

        xf1 = np.maximum(x1, x1gt)
        xf2 = np.minimum(x2, x2gt)
        yf1 = np.maximum(y1, y1gt)
        yf2 = np.minimum(y2, y2gt)

        w = np.maximum(0, xf2-xf1+1)
        h = np.maximum(0, yf2-yf1+1)

        intersections = w*h
        A = (x2-x1+1)*(y2-y1+1)
        B = (x2gt-x1gt+1)*(y2gt-y1gt+1)
        de = (A+B-intersections)

        overlaps.append(intersections/de)
    return np.array(overlaps)

def get_positive_samples(patches, labels, overlaps):
    for i, label in enumerate(labels):
        samples = patches[overlaps[i,:]>0.8]
        l = np.zeros((len(samples),)) + label
        positive_samples.append(np.array(samples))
        positive_labels.append(np.array(l))

def get_negative_samples(patches, overlaps):
    o = np.max(overlaps, axis=0)
    negative_samples.append(np.array(patches[o<0.05]))

### main ###

def extract_train(imgs, classifier=True, recognize=True):
    train = int(len(imgs)*.8)
    train_files = imgs[:train]
    ann_conts = load_ann()
    global positive_labels
    global positive_samples
    global negative_samples
    for i, f in enumerate(train_files):
        img = cv2.imread(f)
        cand_regions, cand_patches, cand_boxes = mser.region_detector(img)
        if cand_patches.shape[0] == 0:
            continue
        true_boxes, true_labels = get_truth(ann_conts[i])
        true_patches = get_patches(img, true_boxes)
        overlaps = calc_overlaps(cand_boxes, true_boxes)
        get_positive_samples(cand_patches, true_labels, overlaps)
        positive_samples.append(true_patches)
        positive_labels.append(true_labels)
        get_negative_samples(cand_patches, overlaps)

    negative_samples = np.concatenate(negative_samples[:len(negative_samples)//6], axis=0)
    negative_labels = np.zeros((len(negative_samples), 1))
    positive_samples = np.concatenate(positive_samples[:len(positive_samples)], axis=0)
    positive_labels = np.concatenate(positive_labels[:len(positive_labels)], axis=0).reshape(-1,1)

    merged_samples = np.concatenate([negative_samples, positive_samples], axis=0)
    merged_labels = np.concatenate([negative_labels, positive_labels], axis=0)

    return merged_samples, merged_labels

    # return positive_samples, positive_labels

def extract_test(imgs, classifier=True, recognize=True):
    test = int(len(imgs)*.8)
    test_files = imgs[test:]
    ann_conts = load_ann()
    global positive_labels
    global positive_samples
    global negative_samples
    positive_samples = []
    positive_labels = []
    negative_samples = []
    for i, f in enumerate(test_files):
        img = cv2.imread(f)
        cand_regions, cand_patches, cand_boxes = mser.region_detector(img)
        if cand_patches.shape[0] == 0:
            continue
        true_boxes, true_labels = get_truth(ann_conts[test])
        true_patches = get_patches(img, true_boxes)
        overlaps = calc_overlaps(cand_boxes, true_boxes)
        get_positive_samples(cand_patches, true_labels, overlaps)
        positive_samples.append(true_patches)
        positive_labels.append(true_labels)
        get_negative_samples(cand_patches, overlaps)
        test+=1

    negative_samples = np.concatenate(negative_samples[:len(negative_samples)//6], axis=0)
    negative_labels = np.zeros((len(negative_samples), 1))
    positive_samples = np.concatenate(positive_samples[:len(positive_samples)], axis=0)
    positive_labels = np.concatenate(positive_labels[:len(positive_labels)], axis=0).reshape(-1,1)

    merged_samples = np.concatenate([negative_samples, positive_samples], axis=0)
    merged_labels = np.concatenate([negative_labels, positive_labels], axis=0)

    # print(merged_labels[-30])
    # cv2.imshow('asdf', merged_samples[-30])
    # cv2.waitKey(0)
    return merged_samples, merged_labels

    # return positive_samples, positive_labels

def write(filename, method, dataset, data, dtype):
    db = h5py.File(filename, method)
    dataset = db.create_dataset(dataset, data.shape, dtype=dtype)
    dataset[:] = data[:]
    db.close()


def compile(DIR='train'):
    files = sorted(os.listdir(DIR), key=lambda x: int(x.split('.')[0]))
    imgs = [os.path.join(DIR, f) for f in files]
    # test = int(len(imgs)*.8)
    # img = cv2.imread(imgs[test])
    # for p in true_patches:
    #     cv2.imshow('sdf', p)
    #     cv2.waitKey(0)
    #
    strain, ltrain = extract_train(imgs)
    stest, ltest = extract_test(imgs)

    # print(strain.shape)
    # print(ltrain.shape)
    # print(stest.shape)
    # print(ltest.shape)

    write('train.hdf5', 'w', 'images', strain, 'uint8')
    write('train.hdf5', 'a', 'labels', ltrain, 'int')
    write('test.hdf5', 'w', 'images', stest, 'uint8')
    write('test.hdf5', 'a', 'labels', ltest, 'int')

if __name__ == '__main__':
    compile()
