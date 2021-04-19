import cv2
import numpy as np

# https://stackoverflow.com/questions/47595684/extract-mser-detected-areas-python-opencv
def region_detector(img):
    hi, wi = img.shape[0], img.shape[1]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_delta=1)
    regions, boxes = mser.detectRegions(grey)
    patches = []
    bbs = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        (x1, y1) = (max(x, 0), max(y, 0))
        (x2, y2) = (min(x+w, wi), min(y+h, hi))
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patches.append(patch)
        bbs.append((y, y+h, x, x+w))

    return regions, np.array(patches), np.array(bbs)

if __name__ == '__main__':
    img = cv2.imread('k.jpg')
    print(len(region_detector(img)))
