import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

IMAGE_PATH = './data/sample_images/'

def listdir_nohidden(dirpath):
    """
    Enumerates all the paths one level deeper than the given directory path.
    """
    for f in os.listdir(dirpath):
        if not f.startswith('.'):
            yield os.path.join(dirpath, f)

def rgb_to_greyscale(imgs):
	greys = []
	for rgb in imgs:
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		greys.append(0.2989 * r + 0.5870 * g + 0.1140 * b)
	return greys

def load_images():
	"""
	Returns list of ndarray images.
	"""
	images = []
	for fp in listdir_nohidden(IMAGE_PATH):
		images.append(ndimage.imread(fp))
	return images

def compare_images(img_list, index, titles):
	fig = plt.figure(figsize=(20,10))
	plt.subplot(1,3,1)
	plt.imshow(img_list[index - 1], cmap='gray')
	plt.title(titles[0])
	plt.axis('off')
	plt.subplot(1,3,2)
	plt.imshow(img_list[index - 1 + 4], cmap='gray')
	plt.title(titles[1])
	plt.axis('off')
	plt.subplot(1,3,3)
	if np.shape(img_list[index - 1]) == np.shape(img_list[index - 1 + 4]):
		difference_image = img_list[index - 1] - img_list[index - 1 + 4]
	else:
		difference_image = img_list[index - 1] - img_list[index - 1 + 4].repeat(4, axis=0).repeat(4, axis=1)
	plt.imshow(difference_image, cmap='gray')
	plt.title(titles[2])
	plt.axis('off')
	plt.gcf().tight_layout()
