import h5py
import numpy as np
import sys
sys.path.append("utils/")
sys.path.append("../")
from color_histograms_utils import feat_extract
from scipy.stats import mode

from project.metrics import *

import matplotlib.pyplot as plt

from PIL import Image

with h5py.File("./data/assorted_images/trucks_and_planes.h5","r") as hf:
    X_cifar = hf["data"][...] / 255.
    y_cifar = hf["labels"][...]

with h5py.File("./data/assorted_images/satellite_images.h5","r") as hf:
    X_satel = hf["data"][...] / 255.
    y_satel = hf["labels"][...]

T = np.load("./data/features/transfer_features.npy")
uga_mean = T.mean(axis=0)
uga_std = T.std(axis=0)

# Train / Test split
np.random.seed(0)
assign = np.random.permutation(len(X_satel))
satel_cutoff = int(0.8 * len(X_satel))
satel_train_assign = assign[:satel_cutoff]
satel_test_assign = assign[satel_cutoff:]

np.random.seed(0)
assign = np.random.permutation(len(X_cifar))
cifar_cutoff = int(0.7 * len(X_cifar))
cifar_train_assign = assign[:cifar_cutoff]
cifar_test_assign = assign[cifar_cutoff:]

def load_trucksplanes_labels():
    return y_cifar[cifar_train_assign][...,None]

def load_satellite_labels():
    return y_satel[satel_train_assign][...,None]

def load_satellite_labels_test():
    return y_satel[satel_test_assign][...,None]

def shuffle_data(H, y):
    np.random.seed(0)
    p = np.random.permutation(len(H))
    return H[p], y[p]

def extract_trucksplanes_histograms(bins, use_hsv):
    X = X_cifar[cifar_train_assign]

    H = []
    for i, img in enumerate(X):
        if (i + 1) % 500 == 0:
            print("Extracted {} of {} histograms".format(i+1,len(X)))
        h = feat_extract(img, bins=bins, use_hsv=use_hsv)
        H.append( h )

    print("Extracted {} of {} histograms".format(len(X),len(X)))
    print("Done!")

    H = np.row_stack(H)
    H = np.column_stack([H,np.ones(len(H))])
    return H

def extract_satellite_histograms(bins, use_hsv):
    X = X_satel[cifar_train_assign]

    H = []
    for i, img in enumerate(X):
        if (i + 1) % (len(X)/10) == 0:
            print("Extracted {} of {} histograms".format(i+1,len(X)))
        h = feat_extract(img, bins=bins, use_hsv=use_hsv)
        H.append( h )

    print("Extracted {} of {} histograms".format(len(X),len(X)))
    print("Done!")

    H = np.row_stack(H)
    H = np.column_stack([H,np.ones(len(H))])
    return H

#def extract_imagenet_features():
#    H = np.load("./data/features/imagenet_features.npy")
#    return H[satel_train_assign]
#
#def extract_nightlights_features():
#    H = np.load("./data/features/nightlights_features.npy")
#    return H[satel_train_assign]
#
#def extract_survey_features():
#    H = np.load("./data/features/survey_features.npy")
#    return H[satel_train_assign]

def extract_uganda_features():
    H = np.load("./data/features/transfer_features.npy")
    H -= uga_mean
    H /= (uga_mean + 1e-8)
    H = np.column_stack([H,np.ones(len(H))])
    return H[satel_train_assign]

def extract_uganda_features_test():
    H = np.load("./data/features/transfer_features.npy")
    H -= uga_mean
    H /= (uga_mean + 1e-8)
    H = np.column_stack([H,np.ones(len(H))])
    return H[satel_test_assign]

def check_training_progress(weights, epoch, epochs):
    if epoch in np.linspace(10,epochs + 10,10).astype("int32"):
        print("Epoch: {}/{}".format(epoch,epochs))
    exploded = np.isnan(np.sum(weights))
    if exploded:
        s = "Your model exploded! "
        s += "Try a smaller learning or regularization rate"
        print(s)
    return exploded

def compute_final_results(hyperparameters, models, data=None, labels=None, use_satellite= False):
    if use_satellite:
        X_t = extract_uganda_features()
        y_t = load_satellite_labels()
        X_v = extract_uganda_features_test()
        y_v = load_satellite_labels_test()
    else:
        assert (data is not None) and (labels is not None)
        cutoff = int(0.7 * len(data))
        X_t = data[:cutoff]
        y_t = labels[:cutoff]
        X_v = data[cutoff:]
        y_v = labels[cutoff:]

    num_features = X_t.shape[-1]
    print("Retraining model on ALL training data")
    trained_models = []
    for model_class in models:
        model = model_class(
                num_features,
                hyperparameters["learning_rate"],
                hyperparameters["regularization_rate"],
                hyperparameters["batch_size"],
                hyperparameters["epochs"]
                )
        model.train(X_t,y_t)
        acc_train = compute_accuracy(model, X_t, y_t)
        acc_test = compute_accuracy(model, X_v, y_v)

        print("TRAINING RESULTS: ")
        compute_all_scores(model, X_t, y_t)

        print("TESTING RESULTS: ")
        compute_all_scores(model, X_v, y_v)

        trained_models.append(model)

    return trained_models

def get_locs():
    with open("./data/survey_data/uga_2011_locs.txt","r") as f:
        lines = f.readlines()
    lines = [list(map(float,(line.strip()).split(' '))) for line in lines]
    return np.array(lines)

def rotate(x, t):
    theta = np.radians(t)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(np.matrix('{} {}; {} {}'.format(c, -s, s, c)))
    return np.dot(x, R)

def change_range(x, newMin, newMax):
    #oldMax = x.max(); oldMin = x.min()
    oldMin = np.array([x[:,0].min(),x[:,1].min()])
    oldMax = np.array([x[:,0].max(),x[:,1].max()])

    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    y = (((x - oldMin) * newRange) / oldRange) + newMin

    #y = newMax - y
    return y

def uganda_map(trained_models):
    k = 25
    f, axs = plt.subplots(1,len(trained_models)+1, figsize=(k, k/4))
    img = Image.open("./sample_images/ugout.gif")
    r, c = np.array(img).shape
    LOCS = get_locs()
    LOCS = rotate(LOCS, 95)
    LOCS = change_range(LOCS, np.array([60.,60.]), np.array([500.,500.]))

    X_t = extract_uganda_features()
    X_v = extract_uganda_features_test()

    y_t = load_satellite_labels()
    y_v = load_satellite_labels_test()
    X = np.row_stack([X_t, X_v])
    y = np.concatenate([y_t, y_v])
    for ax in axs:
        ax.imshow(img)

    for i, model in enumerate(trained_models):
        i += 1
        pred = np.squeeze(model.predict(X))
        C = np.column_stack([pred, np.zeros_like(pred), np.zeros_like(pred)])
        axs[i].scatter(LOCS[:,0], LOCS[:,1], c= C)

        axs[i].set_title(model.name)
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

    C = np.column_stack([y, np.zeros_like(y), np.zeros_like(y)])
    axs[0].scatter(LOCS[:,0],LOCS[:,1],c= C)
    axs[0].set_title("Labels")
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    plt.show()

if __name__ == "__main__":
    uganda_map([None, None, None])

