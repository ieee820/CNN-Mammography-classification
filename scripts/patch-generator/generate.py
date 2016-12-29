#!/usr/bin/env python
import os
import sys
import subprocess
import numpy
import glob
import logging
import argparse
import cv2
import re
import bwlabel
import contextlib

"""
Todo
    Gauss or scale patch!
    Different saturation levels

    # Zoom nivaer
    # Normals: generate depending on right, left, MLO or CC
    # Normals: Don't include to much black or intensity?
    Cancers: Generate larger set from overlay
    Cancers: Use info about core
"""


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield 
    numpy.set_printoptions(**original)

def showImage(image):
    cv2.imshow('image',image)
    key = cv2.waitKey(0)
    if key == 27: sys.exit()

def showImageFile(file):
    image = cv2.imread(file)
    showImage(image)

def readBoundary(file):
    f = open(file, 'r')
    overlay = f.read()
    #print overlay
    lesionsCount = int(re.match(r'TOTAL_ABNORMALITIES\s(\d*)', overlay).group(1))
    contours = []

    boundary = re.findall(r'BOUNDARY[\n\r]*([\d\s]*)#', overlay, re.M)
    for i in range(0,lesionsCount):
        contour = filter(None, boundary[i].split(' '))
        contours.append(contour)

    return contours

def buildContourFromOverlay(boundary, shape, preview=False):
    pos = [int(boundary[1]), int(boundary[0])]
    # Add ~4 due to OOB issues, scale 4x due to prev. scaling
    mask = numpy.zeros(numpy.multiply(numpy.add(shape, 1),4), dtype=numpy.uint8)
    try:
        mask[pos[0], pos[1]] = 255
    except:
        pass

    translations = [[-1, 0], [-1,1], [0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
    for i in range(2,len(boundary)):
        pos = numpy.add(pos,translations[int(boundary[i])])
        try:
            mask[pos[0], pos[1]] = 255
        except:
            pass
    
    (cnts, _) = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(cnts[0])
    if preview:
        im = mask.copy()
        cv2.rectangle(im,(x,y),(x+w,y+h),255,20)
        showImage(im)
    #rect = cv2.minAreaRect(cnts[0])
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    #cv2.drawContours(mask,[box],0,(0,0,255),2)
    return numpy.divide([x,y,w,h],4)

def cropPatch(image, box, minPatchSize, preview=False):
    sz = numpy.array(image.shape)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    while w < minPatchSize:
        w = w+2 if x+w+1 < sz[1] else w
        x = x-1 if x >= 1 else x
    while h < minPatchSize:
        h = h+2 if y+h+1 < sz[0] else h
        y = y-1 if y >= 0 else y
    subpatch = image[y:y+h, x:x+w]
    if preview: showImage(subpatch)
    if (numpy.array(subpatch.shape) < numpy.array([64,64])).all():
        print 'TO SMALL', subpatch.shape
        sys.exit(1)
    return subpatch

def getValidPatch(image, patchSize, patchesPerImage=1, preview=True):
    # Preprocess
    s_image = cv2.GaussianBlur(image,(101,101),0)
    s_image = cv2.normalize(s_image, None, 0, 255, cv2.NORM_MINMAX)
    s_image = s_image.astype(numpy.uint8, copy=False)
    _,threshold = cv2.threshold(s_image,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find largest region/component
    (cnts, _) = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, pts =[c], color=1)
    if preview: showImage(image * mask)
    # Find valid patches
    patches = []
    sz = image.shape
    for i in xrange(patchesPerImage):
        y = numpy.random.randint(patchSize/2,sz[0]-patchSize/2)
        x = numpy.random.randint(patchSize/2,sz[1]-patchSize/2)
        while mask[y,x] == 0:
            y = numpy.random.randint(patchSize/2, sz[0]-patchSize/2)
            x = numpy.random.randint(patchSize/2, sz[1]-patchSize/2)
        subim = image[y-patchSize/2:y+patchSize/2, x-patchSize/2:x+patchSize/2]
        if preview: showImage(subim)
        patches.append(subim)
    return patches

def getLesionSize(mat): # OLD
    pos = [int(mat[1]), int(mat[0])]
    ext = [pos[0], pos[0], pos[1], pos[1]]

    #translations = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    #translations = [[1, 0], [1, 1], [0, 1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]]
    translations = [[-1, 0], [-1,1], [0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
    #translations = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 1], [-1, 1]]

    for o in range(2,len(mat)):
        pos = numpy.add(pos, translations[int(mat[o])])
        ext[0] = min(ext[0], pos[0])
        ext[1] = max(ext[1], pos[0])
        ext[2] = min(ext[2], pos[1])
        ext[3] = max(ext[3], pos[1])

    return ext

def processLimits(image, lesionLimits, preview=True, size=32): ## OLD
    sz = size/2
    images = []
    for limits in lesionLimits:
        if preview:
            cv2.circle(image,(limits[1],limits[0]),sz,(255,255,255))
            showImage(cv2.resize(image, (0,0), fx=0.5, fy=0.5))
        subimage = image[limits[0]-sz:limits[0]+sz, limits[1]-sz:limits[1]+sz]
        if subimage.shape[0] >= size and subimage.shape[1] >= size:
            images.append(subimage)
        else:
            print 'rejected image', subimage.shape
    return images

def processLesions(image, lesionLimits, preview=True, size=32): # OLD
    images = []
    for l in lesionLimits:
        while l[1] - l[0] < size:
            l[0] -= 1
            l[1] += 1 
        while l[3] - l[2] < size:
            l[2] -= 1
            l[3] += 1 
        if preview:
            image[l[0]:l[1], l[2]:l[3]] = 0
            showImage(cv2.resize(image, (0,0), fx=0.5, fy=0.5))
        subimage = image[l[0]:l[1], l[2]:l[3]]
        #print l[0], l[1], l[2], l[3]
        if subimage.shape[0] >= size and subimage.shape[1] >= size:
            images.append(subimage)
        else:
            print 'rejected image', subimage.shape
    return images
def getOverlayData(file):# OLD
    f = open(file, 'r')
    overlay = f.read()
    #print overlay
    lesionsCount = int(re.match(r'TOTAL_ABNORMALITIES\s(\d*)', overlay).group(1))
    lesionLimits = []

    boundary = re.findall(r'BOUNDARY[\n\r]*([\d\s]*)#', overlay, re.M)
    #print boundary
    for i in range(0,lesionsCount):
        limits = filter(None, boundary[i].split(' '))
        #lesionLimits.append([int(limits[1]),int(limits[0])])
        #lesionLimits.append(getLesionSize(limits))
        new_limits = getLesionSize(limits)
        #centers = [(new_limits[0] + new_limits[1])/2, (new_limits[2] + new_limits[3])/2]
        #lesionLimits.append(centers)
        lesionLimits.append(new_limits)

    return lesionLimits

def getIcsData(path):
    file = glob.glob(os.path.join(path, "*.ics"))[0]
    f = open(file, 'r')
    digitizer = re.search(r'^DIGITIZER (\S*).*$', f.read(), re.M).group(1)
    letter = os.path.basename(file)[0]
    return [letter, digitizer]

def normalizeDigitizer(image, path):
    """
        'A' and DBA: OD = ( log(GL) - 4.80662 ) / -1.07553 
        'A' and HOWTEK: OD = 3.789 - 0.00094568 * GL
        'B' or 'C' and LUMISYS:  OD = ( GL - 4096.99 ) / -1009.01
        'D' and HOWTEK: OD = 3.96604095240593 + (-0.00099055807612) * (pixel_value)
    """
    sat = 3.5 #Saturation determines the largest optical density allowed
    letter, digitizer = getIcsData(os.path.dirname(path))
    if letter == 'A' and digitizer == "DBA":
        image = (numpy.log10(image+1e-16) - 4.80662) / -1.07553
    elif letter == 'A' and digitizer == "HOWTEK":
        image = 3.789 - 0.00094568 * image
    elif (letter == 'B' or letter == 'C') and digitizer == "LUMISYS":
        image = ( image - 4096.99 ) / -1009.01
    elif letter == 'D' and digitizer == "HOWTEK":
        image = 3.96604095240593 + (-0.00099055807612) * image

    return numpy.clip(image, 0, sat) / sat # saturate and normalize to [0,1]

def histogramEqualization(image):
    #image = cv2.resize(image, (int(cols * args.scale), int(rows * args.scale)))
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = numpy.uint8(numpy.round(image))
    image = cv2.equalizeHist(image)
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    print numpy.max(image)
    print numpy.min(image)
    print numpy.mean(image)
    return image


def im2double(im):
    np = numpy
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

BIN = os.path.join(os.path.dirname(__file__), "../", "DDSM")

"""
files = ['../DDSM/cases/normals/normal_08/case4600/D_4600_1.RIGHT_MLO.png'] + files
files = ['../DDSM/cases/cancers/cancer_01/case0001/C_0001_1.RIGHT_CC.png'] + files
files = ['../DDSM/cases/normals/normal_11/case1955/A_1955_1.RIGHT_CC.png'] + files
files = ["../DDSM/cases/normals/normal_01/case0002/A_0002_1.LEFT_CC.png"] + files
files = ["../DDSM/cases/normals/normal_02/case0200/A_0200_1.LEFT_CC.png"] + files
"""

def scanFolder(files, isOverlay):
    missing_files = 0
    generated_files = 0

    w_m = 0
    h_m = 0
    #for i in xrange(50):
    for i in xrange(len(files)):
        file = files[i]
        ov_file = file

        if isOverlay: file = file.replace("OVERLAY","png")
        print file
        #print getOverlayData(file)
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED);
        if image is None:
            missing_files += 1
        else:
            image = image.astype(numpy.float64, copy=False)
            #image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
            #image *= 0.1 / numpy.median(image)
            #image = cv2.blur(image,(20,20))
            #image *= 255.0/numpy.max(image)
            #image = image.astype(numpy.uint8, copy=False)
            image = normalizeDigitizer(image, file)
            image = 1 - image
            image = image.astype(numpy.float32, copy=False)
            #showImage(cv2.resize(image, (0,0), fx=0.5, fy=0.5))

            patchSize = 128
            minPatchSize = 128
            patchScale = 0.5
           
            if isOverlay:
                dataset = 'cancers'         
                """lesionLimits = getOverlayData(ov_file)
                lesionLimits = (numpy.array(lesionLimits) * 0.25).astype(numpy.int32)
                #centers = lesionLimits
                patches = processLesions(image, lesionLimits, False, minPatchSize)
                #print len(patches)"""
                patches = []
                boundaries = readBoundary(file.replace(".png", ".OVERLAY"))
                for boundary in boundaries:
                    box = buildContourFromOverlay(boundary, image.shape)
                    w_m += (1.0 * box[2])/len(files)
                    h_m += (1.0 * box[3])/len(files)
                    patch = cropPatch(image, box, minPatchSize)
                    patches.append(patch)
                
            else:
                dataset = 'normals'
                # Pick random patch(es)
                patches = getValidPatch(image, patchSize, 1, False)
                    

            """old_cent = centers.copy()
            slides = [-4, -2, 2, 4]
            for aug1 in slides:
                for aug2 in slides:
                    for cent in old_cent:
                        centers = numpy.append(centers,[cent + [aug1,aug2]],0)"""

            
            #print len(patches)

            for k in xrange(len(patches)):
                newName = os.path.basename(file).replace('.png', str(k) + '.png')
                patch = patches[k]*(2**16 - 1)
                rows, cols = patch.shape
                patch = cv2.resize(patch, (int(cols * patchScale), int(rows * patchScale)), interpolation = cv2.INTER_AREA)
                patch = patch.astype(numpy.uint16)
                cv2.imwrite(os.path.join(os.path.dirname(__file__), 'dataset', dataset, newName), patch)
                generated_files += 1

    print 'mean w', w_m
    print 'mean h', h_m
    print 'Missing files:', missing_files
    print 'Generated files:', generated_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float)
    args = parser.parse_args()

    #numpy.random.shuffle(files)
    print 'Generating cancers set'
    files =  glob.glob(os.path.join(BIN, "cases", "cancers", "*/*/*.OVERLAY"))
    #files = [file[0]]
    #files = ['../DDSM/cases/cancers/cancer_03/case1016/A_1016_1.LEFT_CC.OVERLAY']
    scanFolder(files, True)

    print 'Generating normals set'
    files =  glob.glob(os.path.join(BIN, "cases", "normals", "*/*/*.png"))
    scanFolder(files, False)

    if 0:
        file = files[0].replace(".OVERLAY", ".png")
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED);
        image = image.astype(numpy.float64, copy=False)
        image = normalizeDigitizer(image, file)
        image = 1 - image
        #showImage(image)
        patchSize = 128
        minPatchSize = 64
        patchScale = 1

