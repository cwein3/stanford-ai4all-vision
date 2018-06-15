import numpy as np
import matplotlib.pyplot as plt
import h5py

N = 8
colorband = np.arange(0,N*N)[None,...]

array_0 = np.ones(2 * N, dtype=np.uint8)
array_0[::2] = 0

array_1 = np.ones(2 * N, dtype=np.uint8)
array_1[::3] = 0

array_2 = np.ones(2 * N, dtype=np.uint8)
array_2[::4] = 0

# generate rectangles
rect_0 = np.ones(2 * 5)
rect_0[::2] = 0
rect_0 = rect_0.reshape(2,5)

rect_1 = np.ones(3 * 4)
rect_1[::2] = 0
rect_1 = rect_1.reshape(4,3)

rect_2 = np.ones(2 * 2)
#rect_2[::2] = 0
rect_2[0] = 0
rect_2[-1] = 0
rect_2 = rect_2.reshape(2,2)

rect_3 = np.ones(5,)
rect_3[::2] = 0
rect_3 = rect_3[None,...]

rect_w_col = np.zeros((5,5))
col = np.zeros(5)
col[::2] = 1.
rect_w_col[:,3] = col

def display_all_rectangles(shapes = None):
    f, axs = plt.subplots(1,4)
    axs[0].matshow(rect_0, cmap="Greys")
    axs[1].matshow(rect_1, cmap="Greys")
    axs[2].matshow(rect_2, cmap="Greys")
    axs[3].matshow(rect_3, cmap="Greys")

    if shapes is not None:
        axs[0].set_xlabel(shapes[0])
        axs[1].set_xlabel(shapes[1])
        axs[2].set_xlabel(shapes[2])
        axs[3].set_xlabel(shapes[3])

    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

def display_array(x, ax=None, column=False):
    if x.ndim == 1:
        if column:
            x = x[...,None]
        else:
            x = x[None,...]

    if ax is None:
        plt.matshow(x,cmap="Greys")
    else:
        ax.matshow(x,cmap="Greys")

def display_arrays(x):
    names = ["Array 0 (x0)", "Array 1 (x1)", "Array 2 (x2)",
            "Concatenated Array (x_concatenated)"]
    if x[-1] is not None:
        f, axs = plt.subplots(len(x),1)
    else:
        f, axs = plt.subplots(len(x)-1,1)

    for i, ax in enumerate(axs):
        ax.set_title(names[i])
        #ax.matshow(x[i][None,...],cmap="Greys")
        display_array(x[i], ax=ax)

    if len(axs) == 3:
        print("x_concatenated needs to be defined ...")

    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

def display_images(x):
    names = ["Image 0 (x0)", "Image 1 (x1)",
            "Concatenated Image (x_concatenated)"]

    if x[-1] is not None:
        f, axs = plt.subplots(1, len(x))
    else:
        f, axs = plt.subplots(1, len(x)-1)

    for i, ax in enumerate(axs):
        ax.set_title(names[i])
        ax.imshow(x[i])

    if len(axs) == 2:
        print("x_concatenated needs to be defined ...")

    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()


def display_images_in_grid(xr, xg, xb, xp, xy, xt):
    f, axs = plt.subplots(2,3)
    axs[0,0].imshow(xr)
    axs[0,1].imshow(xg)
    axs[0,2].imshow(xb)

    axs[1,0].imshow(xp)
    axs[1,1].imshow(xy)
    axs[1,2].imshow(xt)

    plt.show()

def load_faces_data():
    with h5py.File("./data/assorted_images/att_faces.h5","r") as hf:
        data = hf["data"][...]
    return data

def load_planes_data():
    with h5py.File("./data/assorted_images/trucks_and_planes.h5","r") as hf:
        data = hf["data"][...]
    return data

def display_color_image_and_shape():
    data = load_planes_data()
    x = data[-100]

    f, axs = plt.subplots(1,4)
    axs[0].imshow(x)
    axs[0].set_xlabel(x.shape)

    xr = x[:,:,0]
    xg = x[:,:,1]
    xb = x[:,:,2]
    axs[1].matshow(xr, cmap="Reds")
    axs[1].set_xlabel(xr.shape)
    axs[1].set_title("Red Channel")

    axs[2].matshow(xg, cmap="Greens")
    axs[2].set_xlabel(xg.shape)
    axs[2].set_title("Green Channel")

    axs[3].matshow(xb, cmap="Blues")
    axs[3].set_xlabel(xb.shape)
    axs[3].set_title("Blue Channel")

    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

