import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from numba import njit, jit, prange
from img2vec import get_features
from distances import metrics
import pickle
import random

with open("data/f30k_features.pkl", "rb") as f:
    all_files = pickle.load(f)

print("Loaded %i feature vectors" % len(all_files))
excluded_keys = []
print(f"Excluded keys: {excluded_keys}")

for entry in all_files:
    for key in entry["features"]:
        entry["features"][key] = entry["features"][key] / np.linalg.norm(
            entry["features"][key])

#selected_image = random.choice(glob("data/f30k/*"))
selected_image = "data/f30k/6979052552.jpg"
print("Selected image: %s" % selected_image)
query = cv2.imread(selected_image)
query_f = get_features(query)
query_features = np.concatenate([(query_f[key]) / np.linalg.norm(query_f[key])
                                 for key in query_f
                                 if key not in excluded_keys])


def apply_metric(metric, query, db):
    distances = []
    for entry in db:
        df = np.concatenate([
            entry["features"][key] for key in entry["features"]
            if key not in excluded_keys
        ])
        distances.append({
            "name": entry["filename"],
            "distance": metric(query, df)
        })
    return distances


plots = []
for metric_name, metric_fn in metrics:
    print("Applying %s metric..." % metric_name)
    distances = apply_metric(metric_fn, query_features, all_files)
    distances = sorted(distances,
                       key=lambda x: x["distance"],
                       reverse=True if metric_name in ["Pearson"] else False)
    fig, ax = plt.subplots(1, 10, figsize=(30, 6))
    ax[0].imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Query")
    ax[0].axis("off")
    # if first image is the query, remove it
    if distances[0]["name"] == selected_image.split("/")[-1].split(".")[0]:
        distances = distances[1:]
    for i in range(0, 9):
        img = cv2.imread("data/f30k/%s.jpg" % distances[i]["name"])
        ax[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i + 1].set_title(
            f"{distances[i]['name']}, distance: {distances[i]['distance']:.4f}"
        )
        ax[i + 1].axis("off")
    fig.tight_layout()
    fig.suptitle(f"{metric_name} Distance", fontsize=24)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plots.append(
        image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3, )))

# Text to be added on top
text = ", ".join([key for key in query_f.keys() if key not in excluded_keys])

vstacked = np.vstack(plots)
vstacked = cv2.cvtColor(vstacked, cv2.COLOR_BGR2RGB)
# add blank space on top
vstacked = np.vstack(
    [np.ones((50, vstacked.shape[1], 3), dtype=np.uint8) * 255, vstacked])
vstacked = cv2.putText(vstacked, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite("results.png", vstacked)
