import cv2
import numpy as np
import faiss
import pickle
from img2vec import img2vec
import sys

with open("data/f30k_descriptors.pkl", "rb") as f:
    descriptors = pickle.load(f)

def cluster_centroids(descriptors):
    desc = np.concatenate(descriptors, axis=0).astype(np.float32)
    kmeans = faiss.Kmeans(desc.shape[1],
                          200,
                          niter=50,
                          nredo=3,
                          verbose=True,
                          gpu=True,
                          seed=42)
    kmeans.train(desc)
    return kmeans

kmeans = cluster_centroids(descriptors)

if sys.argv[1]:
    img = cv2.imread(sys.argv[1])
    vec = img2vec(img)
    vec = vec.astype(np.float32)
    vec = vec.reshape(vec.shape[0], -1)
    centroids, distances = kmeans.index.search(vec, 1)

    print(centroids)