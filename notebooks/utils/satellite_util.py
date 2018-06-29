"""
KEY = AIzaSyA7iLNu_ZPVjG56IB-WM1EM4zrZE9XFCQU
"""
from PIL import Image
import io
#from osgeo import gdalnumeric, gdal, osr
import matplotlib.pyplot as plt
import urllib.request
import os
import numpy as np
import h5py

import scipy.misc as misc

plt.ion()

KEY = "AIzaSyA7iLNu_ZPVjG56IB-WM1EM4zrZE9XFCQU"
with h5py.File("./data/assorted_images/satellite_images.h5","r") as hf:
    images = hf["data"][...]
    labels = hf["labels"][...]
    n = 5
    images_0 = images[labels == 0][:n]
    images_1 = images[labels == 1][:n]

    images = np.concatenate([images_0,images_1],axis=0)
    np.random.seed(1)
    p = np.random.permutation(n * 2)
    images = images[p]
    labels = np.concatenate([np.zeros(n), np.ones(n)])[p]

def display(guesses = None):
    f, axs = plt.subplots(2,n, figsize=(20, 10))
    index_to_label = {}
    for i, ax in enumerate(range(2 * n)):
        ax = axs[i % 2, i // 2]
        index = str(i // 2 + (n * (i % 2)))

        img = images[i]
        # decorate axes
        if guesses is None:
            ax.set_title(index)
        else:
            y_true = int(labels[i])
            y_pred = int(index in guesses.split(","))
            ax.set_title("{} \n (TRUTH: {}; YOUR GUESS: {})".format(
                index,
                y_true,
                y_pred)
                )

            if y_true == y_pred:
                img[:,:,0] = (img[:,:,0] * 0.7).astype('uint8')
                img[:,:,2] = (img[:,:,2] * 0.7).astype('uint8')
            else:
                img[:,:,1] = (img[:,:,1] * 0.7).astype('uint8')
                img[:,:,2] = (img[:,:,2] * 0.7).astype('uint8')

        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        index_to_label[index] = labels[i]
    plt.show()
    return index_to_label

def start_quiz():
    index_to_label = display()
    print("Guess which areas have high poverty. Please enter your answers as a comma separated list")
    print("Example: 0, 1, 2")
    guesses = input("")
    return guesses, index_to_label

def compute_results(guesses, truth):
    _ = display(guesses = guesses)
    guesses = [digit.strip() for digit in guesses.split(",")]
    acc = np.sum([truth[str(g)] for g in guesses]) / float(n)
    print("Your accuracy was: {}%".format(acc * 100.))

def get_coordinate_list(num_locations = 200):
    with open("./data/survey_data/uga_2011_locs.txt","r") as f:
        lines = f.readlines()
    lines = [tuple(map(float,(line.strip()).split(' '))) for line in lines]
    return lines[:num_locations]

def print_coordinates():
    coords = get_coordinate_list()
    for i, coord in enumerate(coords):
        print("site {}: {}, {}".format(i, *coord))

def download_img(lat, lon, zoom = 16, width = 224, height = 224, save = False):
    height += 100 # remove banner at bottom
#    try:
    url_satellite = "https://maps.googleapis.com/maps/api/staticmap?center=" + \
        str(lat) + "," + str(lon) + "&zoom=" + str(zoom) + "&size=" + str(width) + "x" + str(height) + \
        "&sensor=false&maptype=satellite&key=" + KEY
    resource = urllib.request.urlopen(url_satellite)
    f_satellite = io.BytesIO(resource.read())
    img_satellite = Image.open(f_satellite)
    rgb = img_satellite.convert('RGB')
    rgb = rgb.crop((0, 50, width, width + 50))
    savedir = "./data/satellite_images"
    i = len(os.listdir(savedir))
    filename = "img_{}.jpg".format(i)
    if save:
        print("Saving: {}".format(filename))
        misc.imsave(os.path.join(savedir, filename), rgb)
    return np.array(rgb)


