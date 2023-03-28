import cv2
import numpy as np
import glob
from multiprocessing import Pool, TimeoutError
import pickle
from pathlib import Path

nfeatures = 100

def img2vec(img):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, des_sift = sift.detectAndCompute(gray, None)
    _, des_orb = orb.detectAndCompute(gray, None)
    if des_sift is None:
        des_sift = np.array([np.zeros(128)])
    if des_orb is None:
        des_orb = np.array([np.zeros(32)])
    return {
        "sift": des_sift,
        "orb": des_orb,
    }


def multiprocessing_child(img_path):
    img = cv2.imread(img_path)
    des = img2vec(img)
    if des["sift"] is None and des["orb"] is None:
        print(f"Image {img_path} has no descriptors!")
        return None
    return {
        "filename": Path(img_path).stem,
        "descriptors": des,
        "class": Path(img_path).parent.name
    }


if __name__ == '__main__':
    queue = glob.glob("../landscapes/train/**/*.jpeg")
    results = []
    print(f"Number of features: {nfeatures}")
    with Pool(processes=13) as pool:
        promises = pool.imap_unordered(multiprocessing_child,
                                       queue,
                                       chunksize=100)
        for promise in promises:
            if promise is not None:
                results.append(promise)
            print(f"Processed {len(results)} / {len(queue)} images")
    with open("data/buildings_descriptors.pkl", "wb") as f:
        pickle.dump(results, f)
