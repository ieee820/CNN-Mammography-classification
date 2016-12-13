#!/usr/bin/python

from os import listdir
from os import path, sep
import sys, random

percent_test = 20
balanced = False

if len(sys.argv) > 1:
	percent_test = int(sys.argv[1])
if len(sys.argv) == 3 and str(sys.argv[2]) == "balanced":
	balanced = True

LEVEL_UP = '..' + sep
TOP_FOLDER = LEVEL_UP + 'data' + sep
CANCER_FOLDER = TOP_FOLDER + 'cancers' + sep;
NORMAL_FOLDER = TOP_FOLDER + 'normals' + sep;
PNG_FOLDER = 'PNGFiles' + sep



cancer_set = []
cancer_paths = listdir(CANCER_FOLDER)
cancer_paths = [x for x in cancer_paths if not x.find('cancer') == -1]

print('Found %d cancer_xx folders' % len(cancer_paths))

for cancer_path in cancer_paths:
	if not cancer_path.find('cancer') == -1:
		print('Processing path %s' % cancer_path)

		for case_path in listdir(CANCER_FOLDER + cancer_path):
			if not case_path.find('case') == -1: # remove?
				print('Processing case %s' % case_path)
				overlay_files = listdir(CANCER_FOLDER + cancer_path + sep + case_path);
				overlay_files = [x for x in overlay_files if not x.find('.OVERLAY') == -1]
				print('Found %d overlay files' % len(overlay_files))
				for overlay_file in overlay_files:
					print('Processing overlay file %s' % overlay_file)
					overlay_file = overlay_file.split('.OVERLAY')[0]
					png_file = path.join(CANCER_FOLDER, cancer_path, case_path, PNG_FOLDER, overlay_file + '.png')
					cancer_set.append([path.abspath(png_file), 1])

normal_set = []
normal_paths = listdir(NORMAL_FOLDER)
normal_paths = [x for x in normal_paths if not x.find('normal') == -1]

for normal_path in normal_paths:
	if not normal_path.find('normal') == -1: # remove?
		print('Processing path %s' % normal_path)

		for case_path in listdir(NORMAL_FOLDER + normal_path):
			if not case_path.find('case') == -1:
				print('Processing case %s' % case_path)
				ljpeg_files = listdir(NORMAL_FOLDER + normal_path + sep + case_path);
				ljpeg_files = [x for x in ljpeg_files if not x.find('.LJPEG') == -1]
				print('Found %d ljpeg files' % len(ljpeg_files))
				for ljpeg_file in ljpeg_files:
					print('Processing ljpeg file %s' % ljpeg_file)
					ljpeg_file = ljpeg_file.split('.LJPEG')[0]
					png_file = path.join(NORMAL_FOLDER, normal_path, case_path, PNG_FOLDER, ljpeg_file + '.png')
					normal_set.append([path.abspath(png_file), 0])

nbr_cancer = len(cancer_set)
nbr_normal = len(normal_set)

if balanced:
	if nbr_cancer < nbr_normal:
		nbr_normal = nbr_cancer
	else:
		nbr_cancer = nbr_normal

full_set = cancer_set[0:nbr_cancer] + normal_set[0:nbr_normal]
random.shuffle(full_set)
nbr_train = int(len(full_set) * (100-percent_test)/100)
nbr_test = len(full_set) - nbr_train

print('Opening training set file for print')
f = open('training_set', 'w')
for i in range(0,nbr_train):
	f.write('%s %d\n' % (full_set[i][0], full_set[i][1]))
f.close()

print('Opening test set file for print')
f = open('test_set', 'w')
for i in range(nbr_train + 1,nbr_train + nbr_test):
	f.write('%s %d\n' % (full_set[i][0], full_set[i][1]))
f.close()

print('\n-------------------------\nSuccessful')
print('Processed %d cancer images and %d normal images' % (nbr_cancer,nbr_normal))
if balanced:
	print('Using balanced training set')
print('Using %d training and %d test images' % (nbr_train, nbr_test))

