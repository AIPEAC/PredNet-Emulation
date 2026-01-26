"""
	point of this py:
		to process image data
	kitti:
		a image set
"""

"""
I: imports
"""
# this is a local file
from kitti_settings import *

# regular imports
import os
import numpy as np

# for http usage to get image sets
import requests
import urllib.request

# decode HTML
from bs4 import Beautifulsoup

# read image from file, adjust image size
from imageio import imread

# to save & reload python data structures
import hickle as hkl

# use PIL instead of scipy.misc since the later is deprecated
#from scipy.misc import imresize
from PIL import Image
def imresize(arr, size):
    img = Image.fromarray(arr)
    img_resized = img.resize(size, Image.ANTIALIAS)
    return np.array(img_resized)


"""II: global settings"""
desired_image_size= (128,160)
categories = ['city','residential','road']


"""II.2: validation sets and test sets"""
validation_name1=["2011_09_26_drive_0005_sync"]
val_recordings=[('city',validation_name1[0])]

test_names1=['2011_09_26_drive_0104_sync',
             '2011_09_26_drive_0079_sync',
             '2011_09_26_drive_0070_sync']
test_recordings=[('residential',test_names1[0]),
				 ('road',test_names1[1]),
				 ('city',test_names1[2])]

if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)