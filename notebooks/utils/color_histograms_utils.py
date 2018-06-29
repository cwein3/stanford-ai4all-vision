import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

with h5py.File("./data/assorted_images/trucks_and_planes.h5","r") as hf:
    X_cifar = hf["data"][...] / 255.
    y_cifar = hf["labels"][...]

with h5py.File("./data/assorted_images/satellite_images.h5","r") as hf:
    X_satel = hf["data"][...] / 255.
    y_satel = hf["labels"][...]
    # evenly split classes
    X_satel_0 = X_satel[y_satel == 0]
    X_satel_1 = X_satel[y_satel == 1]
    k = np.minimum(len(X_satel_0), len(X_satel_1))
    X_satel_0 = X_satel_0[:k]
    X_satel_1 = X_satel_1[:k]
    X_satel = np.concatenate([X_satel_0, X_satel_1], axis=0)
    y_satel = np.concatenate([np.zeros(k),np.ones(k)])

np.random.seed(0)
p = np.random.permutation(len(X_cifar))
data_sample = X_cifar[p][:8]

def shuffle_data(H, y):
    np.random.seed(0)
    p = np.random.permutation(len(H))
    return H[p], y[p]

def train_model_and_get_results(H_t, H_v, y_t, y_v):
    if H_t is None:
        print("You forgot to define the training set!")
        return
    if H_v is None:
        print("You forgot to define the testing set!")
        return

    if y_t is None:
        print("You forgot to define labels for the training set!")
        return
    if y_v is None:
        print("You forgot to define labels for the testing set!")
        return

    model = LogisticRegression()
    model.fit(H_t,y_t)
    p_t = model.predict(H_t)
    p_v = model.predict(H_v)

    acc_t = accuracy_score(y_t, p_t)
    acc_v = accuracy_score(y_v, p_v)

    print("training accuracy_cifar: {}; testing accuracy_cifar: {}".format(acc_t, acc_v))

def scal_to_vec(s):
#    s0 = (s & 255)/255.,
#    s1 = ((s >> 8) & 255)/255.,
#    s2 = ((s >> 16) & 255)/255.,
#    return np.array_cifar(s0 + s1 + s2)
    return np.array_cifar([s % 255, (s // 255) % 255, ((s // 255) // 255) % 255]) / 255.

def show_img_and_hist(img, h, C):
    f, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(img)
    axs[1].bar(np.arange(len(h)), h, color= C)
    plt.show()

def compare_hists(h0, h1, bins, use_hsv, use_satellite= False, names=None):
    f, axs = plt.subplots(1,2, figsize=(20,10))
    C = create_colors(bins, use_hsv)
    axs[0].bar(np.arange(len(h0)), h0, color= C, edgecolor="k")
    axs[1].bar(np.arange(len(h1)), h1, color= C, edgecolor="k")

    if names is None:
        if use_satellite:
            names = ["Low Poverty", "High Poverty"]
        else:
            names = ["Airplanes", "Trucks"]

    if len(names) > 0:
        for (name, ax) in zip(names,axs):
            ax.set_title(name)

    y_lo_0, y_hi_0 = axs[0].get_ylim()
    y_lo_1, y_hi_1 = axs[1].get_ylim()
    axs[0].set_ylim(np.minimum(y_lo_0, y_lo_1), np.maximum(y_hi_0, y_hi_1))
    axs[1].set_ylim(np.minimum(y_lo_0, y_lo_1), np.maximum(y_hi_0, y_hi_1))

    plt.show()

def create_colors(bins, use_hsv):
    if use_hsv:
        Ch = np.column_stack([np.linspace(0,1,bins), np.ones(bins) * 0.7,
                np.ones(bins) * 0.5])
        Cs = np.column_stack([np.ones(bins), np.linspace(0,1,bins), 0.5 *
            np.ones(bins)])
        Cv = np.column_stack([np.ones(bins) * 0.2, np.ones(bins) * 0.5,
                np.linspace(0,1,bins)])
        C = hsv_to_rgb(np.row_stack([Ch, Cs, Cv]))
    else:
        C = np.column_stack([np.linspace(0,1,bins), np.zeros(bins),
                np.zeros(bins)])
        C = np.row_stack([C, np.roll(C,1,axis=1), np.roll(C,2,axis=1)])
    return C

def feat_extract(x, bins, use_hsv):
    if use_hsv: # convert an RGB to an HSV function
        x = rgb_to_hsv(x)
    x = x.reshape(-1,3)
    h0, _ = np.histogram(x[:,0], range=(0,1), bins=bins)
    h1, _ = np.histogram(x[:,1], range=(0,1), bins=bins)
    h2, _ = np.histogram(x[:,2], range=(0,1), bins=bins)
    h = np.concatenate([h0,h1,h2])
    return h

def test_image_to_histogram(f):
    img = X_cifar[0]
    bins = 10
    use_hsv = False

    h0 = f(img, 10, False)
    h1 = feat_extract(img, 10, False)

    diff = np.abs(h0 - h1).astype('float32')

    if np.sum(diff) == 0.0:
        C = create_colors(bins, use_hsv)
        msg = "Your histogram looks perfect!"
    else:
        C = np.column_stack([diff, diff, diff])
        C /= np.maximum(C.max(),0.01)

        C = 1. - C
        msg = "Something appears to be wrong with your histogram. "
        msg += "The greatest errors are shown in darker colors. "

    compare_hists(h0, h1, bins, use_hsv, names= ["Your Histogram", "True Histogram"])
    print(msg)

def load_labels(use_satellite= False):
    if use_satellite:
        return y_satel
    else:
        return y_cifar

def extract_all_histograms(bins, use_hsv, use_satellite= False):
    if use_satellite:
        X = X_satel
    else:
        X = X_cifar

    H = []
    C = create_colors(bins, use_hsv)
    for i, img in enumerate(X):
        if (i + 1) % 500 == 0:
            print("Extracted {} of {} histograms".format(i+1,len(X)))
        h = feat_extract(img, bins=bins, use_hsv=use_hsv)
        H.append( h )

    print("Extracted {} of {} histograms".format(len(X),len(X)))
    print("Done!")

    H = np.row_stack(H)
    return H

