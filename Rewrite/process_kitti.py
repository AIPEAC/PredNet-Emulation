"""
	point of this py:
		to process image data
	kitti:
		a image set
"""


"""I: imports"""
# this is a local file
from unicodedata import category
from Rewrite.kitti_settings import *

# regular imports
import os
import numpy as np

# for http usage to get image sets
import requests
import urllib.request

# decode HTML
from bs4 import BeautifulSoup

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


"""III: download raw data from KITTI website"""
def download_data():
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): 
        os.mkdir(base_dir)
    for category in categories:
        url="http://www.cvlibs.net/datasets/kitti/raw_data.php?type="+category
        response = requests.get(url)
        soup = BeautifulSoup(response.content)
        
        # h3: the 3rd level header tag in a HTML file
        drive_list_original = soup.find_all('h3')
        
        drive_list = []
        for drive in drive_list_original:
            drive_list.append(drive.text[:drive.text.find(' ')])
        print(category)
        category_dir = base_dir + category + '/'
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        #enumerate: return a tuple (index, element) when iterating
        for idx, drive in enumerate(drive_list):
            print(idx+1,'/'+len(drive_list)+': downloading '+drive)
            url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/" + drive + "/" + drive + "_sync.zip"
            urllib.request.urlretrieve(url, filename=category_dir + drive + "_sync.zip")


"""IV: extract data from zip files"""
# unzip images
def extract_data():
    for category in categories:
        category_dir = os.path.join(DATA_DIR, 'raw/', category + '/')
        zip_files = list(os.walk(category_dir, topdown=False))[-1][-1]
        for file in zip_files:
            print('Extracting '+file)
            
            # get the first 10 characters; f[:-4] remove '.zip'
            spec_folder = file[:10] + '/' + file[:-4] + '/image_03/data/'
            """
            	e.g.:
            		file = '2011_09_26_drive_0005_sync.zip'
					spec_folder = '2011_09_26/2011_09_26_drive_0005_sync/image_03/data/'
            """
            
            # use cmd line to unzip
            command = 'unzip -qq ' + category_dir + file + ' ' + spec_folder + ' -d ' + category_dir + file[:-4]
            os.system(command)


"""V: process data, save them in different set"""
def process_data():
	splits = {
		'train': [],
		'val': val_recordings,
		'test': test_recordings
	}
	not_train_recordings = val_recordings + test_recordings
	for category in categories:
		category_dir = os.path.join(DATA_DIR, 'raw/', category + '/')
		folder_list = list(os.walk(category_dir, topdown=False))[-1][-2]
		splits['train'] += [(category, folder) 
                      for folder in folder_list 
                      if (category, folder) not in not_train_recordings]
	
	for split in splits:
		print('Processing '+split+' set')
		image_list = []
		source_list = []
		for (category, folder) in splits[split]:
			print('  Category: '+category+', Folder: '+folder)
   
			""" 				
				DATA_DIR: The root directory for storing data.
				'raw/': Subdirectory for raw data.
				category: The name of the category.
				folder: The folder name of the current recording.
				folder[:10]: Extracts the first 10 characters of the folder name 
    				(usually the date part, e.g., '2011_09_26').
				'image_03/data/': The specific path for image data.
				Finally, image_dir is the directory path for the image data of the current recording.
			"""
			image_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/')

			image_files = list(os.walk(image_dir, topdown=False))[-1][-1]
   
			# add all image files to list
			image_list += [image_dir + file for file in image_files]

			# add source info for each image, 
   			# e.g. 'city-2011_09_26_drive_0005_sync' will occur len(image_files) times
			source_list += [category + '-' + folder] * len(image_files)

		print( 'Creating ' + split + ' data: ' + str(len(image_list)) + ' images')
		
		# initialize numpy array full of zeros
		"""the dimentions are:
			number of images,
			desired height,
			desired width,
			number of color channels (3 for RGB)
			np.uint8: data type for each pixel value (unsigned 8-bit integer)
   		"""
		saved_image_processed_dimensions = (len(image_list), desired_image_size[0], desired_image_size[1], 3)
		saved_image_processed = np.zeros(saved_image_processed_dimensions, dtype=np.uint8)
		for index, image_file in enumerate(image_list):
			
			# imread: read image from file, 
   			# returns a numpy array which dimensions are (height, width, color channels)
			image = imread(image_file)

			# save processed image to numpy array
			saved_image_processed[index] = process_image(image, desired_image_size)

		# hlk.dump: save data structure to file
		hkl.dump(saved_image_processed, os.path.join(DATA_DIR, 'processed_' + split + '.hkl'))
		hkl.dump(source_list, os.path.join(DATA_DIR, 'source_' + split + '.hkl'))


"""VI: resize and crop"""
def process_image(image, desired_size):
    
    # ratio is equal to desired height / original height
	ratio = float(desired_size[0]) / image.shape[0]
 
	image_resized = imresize(image, (desired_size[0], int(np.round(image.shape[1] * ratio))))
	delete_width_from_both_sides = (image_resized.shape[1] - desired_size[1]) // 2
 
	# crop the image to the desired width
	# height: all rows (:)
	# width: from delete_width_from_both_sides to delete_width_from_both_sides + desired width
	# color channels: all channels (:)
	image_cropped = image_resized[
     	:, 
        delete_width_from_both_sides : delete_width_from_both_sides + desired_size[1], 
        :]
	return image_cropped

"""VII: main function"""
if __name__ == '__main__':
    download_data()
    extract_data()
    process_data()