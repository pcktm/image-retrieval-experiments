import cv2
import numpy as np
from numba import njit, jit, prange, objmode
from numba.typed import List
from PIL import Image

default_palette = Image.open("data/palette.png")

# Source: https://www.mathworks.com/matlabcentral/fileexchange/46093-color-auto-correlogram
# Translated to Python by me lol
@njit(fastmath=True, cache=True)
def get_n(n, x, y, color, img_no_dither, X, Y):
    valid_vector8n = np.zeros(8 * n)
    positive_count = 0
    total_count = 0
    nbrs_x = np.zeros(8 * n)
    nbrs_y = np.zeros(8 * n)

    nbrs_y[0] = y
    d = 1
    for k in prange(1, 1 + n):
        nbrs_y[k] = y - d
        d = d + 1

    nbrs_y[1 + n:3 * n + 1] = y - n

    d = 0
    for k in prange(3 * n + 1, 5 * n + 1):
        nbrs_y[k] = y - n + d
        d = d + 1

    nbrs_y[5 * n + 1:7 * n + 1] = y + n

    d = 0
    for k in prange(7 * n + 1, 7 * n + 1 + (n - 1)):
        nbrs_y[k] = y + n - d
        d = d + 1

    nbrs_x[0] = x - n

    nbrs_x[1:1 + n] = x - n

    d = 0
    for k in prange(1 + n, 3 * n + 1):
        nbrs_x[k] = x - n + d
        d = d + 1

    nbrs_x[3 * n + 1:5 * n + 1] = x + n

    d = 0
    for k in prange(5 * n + 1, 7 * n + 1):
        nbrs_x[k] = x + n - d
        d = d + 1

    nbrs_x[7 * n + 1:7 * n + 1 + (n - 1)] = x - n

    for i in prange(8 * n):
        if nbrs_x[i] > 0 and nbrs_x[i] <= X and nbrs_y[i] > 0 and nbrs_y[
                i] <= Y:
            valid_vector8n[i] = 1

        else:
            valid_vector8n[i] = 0
    for j in range(8 * n):
        if valid_vector8n[j] == 1:
            data = img_no_dither[int(nbrs_y[j]) - 1][int(nbrs_x[j]) - 1]
            if (data == color):
                positive_count = positive_count + 1
            total_count = total_count + 1
    return positive_count, total_count


@njit(cache=True)
def parallell_helper_correlogram(img, distance_vector: List):
    correlogram_vector = np.array([np.float64(x) for x in range(0)])
    Y, X = img.shape[:2]
    d = len(distance_vector)
    count_matrix = np.zeros((256, d))
    total_matrix = np.zeros((256, d))
    prob_dist = np.zeros((256, d))
    for serial_no in range(d):
        for x in range(X):
            for y in range(Y):
                color = img[y, x]
                positive_count, total_count = get_n(distance_vector[serial_no],
                                                    x, y, color, img, X, Y)
                count_matrix[color, serial_no] += positive_count
                total_matrix[color, serial_no] += total_count
        prob_dist[:, serial_no] = count_matrix[:, serial_no] / (
            1 + total_matrix[:, serial_no])
    for serial_no in range(d):
        correlogram_vector = np.concatenate(
            (correlogram_vector, prob_dist[:, serial_no]))
    return correlogram_vector


def color_auto_correlogram(I, distance_vector=[1, 3], palette=default_palette):
    img_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    image_reduced = im_pil.quantize(colors=256,
                                    method=Image.Quantize.MEDIANCUT,
                                    dither=Image.Dither.NONE,
                                    palette=palette)
    img_no_dither = np.array(image_reduced)
    return parallell_helper_correlogram(img_no_dither, List(distance_vector))


if __name__ == "__main__":
    img = cv2.imread("data/lena.png")
    print(color_auto_correlogram(img))