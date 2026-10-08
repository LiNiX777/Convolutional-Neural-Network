import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from conv import Convolutional_layer
from train import predict
 
 
def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
 
 
def plot_kernels(network, path):
    conv = next(l for l in network if isinstance(l, Convolutional_layer))
    kernels = conv.kernel[:, 0]                      # (8, 3, 3): one input channel
    limit = np.max(np.abs(kernels))                  # same color scale for all, 0 = white
    fig, axes = plt.subplots(1, len(kernels), figsize=(len(kernels) * 1.3, 1.7))
    for i, (ax, k) in enumerate(zip(axes, kernels)):
        ax.imshow(k, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"kernel {i}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Learned 3x3 kernels (red = positive, blue = negative weight)", fontsize=10)
    _save(fig, path)
 
 
def plot_feature_maps(network, x, path):
    """Runs one image through conv + ReLU and shows each kernel's output."""
    out = x
    for layer in network[:2]:                        # conv, then ReLU
        out = layer.forward_pass(out)
    fig, axes = plt.subplots(1, len(out) + 1, figsize=((len(out) + 1) * 1.3, 1.8))
    axes[0].imshow(x[0], cmap="gray")
    axes[0].set_title("input", fontsize=9)
    for i, (ax, fmap) in enumerate(zip(axes[1:], out)):
        ax.imshow(fmap, cmap="viridis")
        ax.set_title(f"kernel {i}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Feature maps: where each kernel responds (bright = strong)", fontsize=10)
    _save(fig, path)
 
 
def predictions(network, x_data):
    return np.array([np.argmax(predict(network, x)) for x in x_data])
 
 
def plot_confusion_matrix(y_true, y_pred, path):
    matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(matrix, cmap="Blues")
    for t in range(10):
        for p in range(10):
            if matrix[t, p]:
                ax.text(p, t, matrix[t, p], ha="center", va="center", fontsize=8,
                        color="white" if matrix[t, p] > matrix.max() / 2 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    ax.set_title(f"Confusion matrix (accuracy {np.mean(y_true == y_pred):.1%})")
    _save(fig, path)
 
 
def plot_misclassified(x_data, y_true, y_pred, path, n=12):
    wrong = np.where(y_true != y_pred)[0][:n]
    if len(wrong) == 0:
        return
    fig, axes = plt.subplots(1, len(wrong), figsize=(len(wrong) * 1.3, 1.9), squeeze=False)
    for ax, i in zip(axes[0], wrong):
        ax.imshow(x_data[i][0], cmap="gray")
        ax.set_title(f"true {y_true[i]}\npred {y_pred[i]}", fontsize=9, color="red")
        ax.axis("off")
    fig.suptitle("Misclassified test images", fontsize=10)
    _save(fig, path)
 
 
def make_all_plots(network, x_test, y_test, folder="results"):
    y_true = np.argmax(y_test.reshape(len(y_test), -1), axis=1)
    y_pred = predictions(network, x_test)
    plot_kernels(network, f"{folder}/kernels.png")
    plot_feature_maps(network, x_test[0], f"{folder}/feature_maps.png")
    plot_confusion_matrix(y_true, y_pred, f"{folder}/confusion.png")
    plot_misclassified(x_test, y_true, y_pred, f"{folder}/misclassified.png")
 
 
if __name__ == "__main__":
    from data import load_mnist, preprocess
    from model import load_model
    _, (x_test, y_test) = load_mnist()
    x_test, y_test = preprocess(x_test, y_test, 10000)
    make_all_plots(load_model("results/model.npz"), x_test, y_test)
    print("plots saved in results/")
