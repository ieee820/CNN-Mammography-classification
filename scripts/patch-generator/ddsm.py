import glob
import os
#import argparse
import cv2
import numpy
from tqdm import *
import tensorflow as tf

"""
	Subtract mean over image!?
	Implement in TF instead: Subtract mean and std over channels !!
	# Verify that Shuffle data between epochs works
"""


cancers_files = glob.glob(
	os.path.join(os.path.abspath(os.path.dirname(__file__)),
	'dataset','cancers','*.png'))
normals_files = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset','normals','*.png'))

print 'Found %d cancers and %d normals files' % (len(cancers_files), len(normals_files))

#parser = argparse.ArgumentParser()
#parser.add_argument("-s","--size", type=float, help='total size of dataset')
#args = parser.parse_args()

# Balance dataset
small = numpy.min([len(cancers_files), len(normals_files)])

data = cancers_files[0:small] + normals_files[0:small]
labels = []
for i in xrange(len(cancers_files[0:small])):
	labels.append([1.0, 0.0])
for i in xrange(len(normals_files[0:small])):
	labels.append([0.0, 1.0])

data = numpy.asarray(data)
labels = numpy.asarray(labels)

# Shuffle
perm = range(0,len(data))
numpy.random.shuffle(perm)
labels = labels[perm]
data = data[perm]

# Divide into training and test set
percent_train = 0.85
lim = int(percent_train * len(data))
data_train = data[:lim];
labels_train = labels[:lim];
data_test = data[lim:];
labels_test = labels[lim:];
print 'Generated %d training and %d test images' % (len(data_train), len(data_test))

patch_size = 64
max_scale = 0.8
#sz = patch_size / 2
def readImage(file):
	img = (cv2.imread(file, cv2.IMREAD_UNCHANGED).astype('float') / ((2**16)-1))

	if not numpy.random.randint(0, 4) == 0:
 		img = cv2.flip(img, numpy.random.randint(-1, 2))

	"""dim_s = numpy.min(img.shape)

	scale_s = (1.0 * patch_size) / dim_s
	scale_s = scale_s if scale_s > max_scale else max_scale

	scale = (scale_s - 1) * numpy.random.ranf() + 1

	img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)"""

	rows, cols = img.shape
	rows_r = numpy.random.randint(0, rows-patch_size+1)
	cols_r = numpy.random.randint(0, cols-patch_size+1)
	img = img[rows_r:rows_r+patch_size, cols_r:cols_r+patch_size]


	return img

def readTestImage(file):
	img = (cv2.imread(file, cv2.IMREAD_UNCHANGED).astype('float') / ((2**16)-1))
	# Return center patch of size patch_size
	y = int(img.shape[0] / 2 - patch_size/2)
	x = int(img.shape[1] / 2 - patch_size/2)
	img = img[y:y+patch_size, x:x+patch_size]
	#img = cv2.resize(img, (0,0), fx=32, fy=32, interpolation = cv2.INTER_AREA)
	return img

# Subtract mean accross pixels
im_mean = 0
im_var = 0
pxl = 0
for file in tqdm(data_train):
	im = (cv2.imread(file, cv2.IMREAD_UNCHANGED).astype('float') / ((2**16)-1))
	im_mean += numpy.mean(im)/len(data_train)
	im_var += numpy.var(im) * im.size
	pxl += im.size

im_std = numpy.sqrt(im_var/pxl)

#im_mean = 0
#im_std = 1

print('Mean: %f, Std: %f' % (im_mean, im_std))

"""x = 0
x_2 = 0
for file in tqdm(data_train):
	im = readImage(file)
	x += im/len(data_train)
	x_2 += numpy.square(im)/len(data_train)
std = numpy.sqrt(x_2 - numpy.square(x))"""
# Normalize to std = 1 accross pixels


# Read size from sample image
size = readImage(data[0]).shape
def getSize():
	return size

def getDataTrainSize():
	return len(data_train)


count = 0
def getTrainBatch(batch_size):
	global count
	global size
	global data_train
	global labels_train
	im = numpy.zeros([batch_size, size[0] * size[1]])
	labs = numpy.zeros([batch_size,2])
	i = 0
	#calc_mean
	while i < batch_size:
		img = readTestImage(data_train[count]).flatten()
		#img -= numpy.mean(img)
		#img -= x.flatten()
		#img /= std.flatten()
		img -= im_mean
		img /= im_std
		im[i,:] = img
		#labs[i,0] = labels_train[count][0]
		#labs[i,1] = labels_train[count][1]
		labs[i,:] = labels_train[count,:]
		count += 1
		i += 1
		if count == len(data_train):
			count = 0
			# Shuffle Dataset
			perm = range(0,len(data_train))
			numpy.random.shuffle(perm)
			labels_train = labels_train[perm]
			data_train = data_train[perm]
			break
	#labs = tf.cast(labs, tf.float32)
	#im = numpy.transpose(im, [2, 1, 0])
	
	#im = numpy.reshape(im, [im.ndarray.shape[0]*im.ndarray.shape[1], im.ndarray.shape[3]])
	return [im, labs]


def getTest():
	im = numpy.zeros([len(data_test), size[0] * size[1]])
	for i in xrange(len(data_test)):
		file = data_test[i]
		img = readTestImage(file).flatten()
		img -= im_mean
		img /= im_std
		im[i,:] = img

	return [im, labels_test]



