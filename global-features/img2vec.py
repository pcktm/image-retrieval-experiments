from concurrent.futures import process
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pywt
from glob import glob
from numba import njit, jit
from correlogram import color_auto_correlogram
from gabor import gabor_transform
from multiprocessing import Pool, TimeoutError
import imutils

def quantize_histogram(img, bins=(16, 8, 8)):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        features.extend(
            cv2.calcHist([hsv_image], [i], None, [bins[i]],
                         [0, 256]).flatten())
    return np.array(features)


def wavelet_transform(img, level=3, family="db3"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dd = pywt.wavedec2(img, family, level=level)
    coeffs = []
    for i in dd:
        if isinstance(i, tuple):
            for j in i:
                coeffs.append(j)
        else:
            coeffs.append(i)
    stds = [np.std(c) for c in coeffs]
    means = [np.mean(c) for c in coeffs]
    return np.concatenate((stds, means))


def standard_RGB(img):
    b, g, r = cv2.split(img)
    return np.std(b), np.std(g), np.std(r)


def mean_RGB(img):
    b, g, r = cv2.split(img)
    return np.mean(b), np.mean(g), np.mean(r)


def get_features(img):
    image = imutils.resize(img, width=256)
    #print("Quantize histogram...")
    quants = quantize_histogram(image)
    #print("Wavelet transform...")
    wavelets = wavelet_transform(image)
    #print("Color auto correlogram...")
    correlogram = color_auto_correlogram(image)
    #print("RGB standard deviation and mean...")
    rgb = np.concatenate((standard_RGB(image), mean_RGB(image)))
    #print("Gabor transform...")
    gabor = gabor_transform(image)
    features = {
        "histogram": quants,
        "wavelets": wavelets,
        "correlogram": correlogram,
        "rgb": rgb,
        "gabor": gabor
    }
    return features

def multiprocessing_child(filename):
    img = cv2.imread(filename)
    features = get_features(img)
    return {
        "filename": Path(filename).stem,
        "features": features,
        "class": Path(filename).parent.name.lower()
    }

if __name__ == "__main__":
    queue = glob("../buildings/*.jpg")
    with Pool(processes=25) as pool:
        results = []
        promises = pool.imap_unordered(multiprocessing_child, queue, chunksize=50)
        for promise in promises:
            results.append(promise)
            print(f"Processed {len(results)} / {len(queue)} images")
    with open("data/buildings_features.pkl", "wb") as f:
        pickle.dump(results, f)
