import cv2
import numpy as np

gabor_filters = []
for scale in [1, 2, 2.5, 3]:
    for theta in np.arange(0, np.pi, np.pi / 6):
        k = cv2.getGaborKernel((60, 60),
                               5.0,
                               theta,
                               6.5,
                               0.5,
                               0,
                               ktype=cv2.CV_32F)
        k /= 1.5 * k.sum()
        kern = cv2.resize(k, (0, 0),
                       fx=scale,
                       fy=scale,
                       interpolation=cv2.INTER_CUBIC)
        # crop a square from the center of the kernel
        kern = kern[int(kern.shape[0] / 2) - 14:int(kern.shape[0] / 2) + 14,
                    int(kern.shape[1] / 2) - 14:int(kern.shape[1] / 2) + 14]
        gabor_filters.append(kern)


def gabor_transform(img):
    local_energies = []
    mean_amplitude = []
    for kern in gabor_filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        local_energies.append(np.sum(fimg**2))
        mean_amplitude.append(np.mean(np.abs(fimg)))
    return np.concatenate((local_energies, mean_amplitude))
