#!/usr/bin/env python
import os
import sys
import re
import subprocess
import numpy
import logging

BIN = os.path.join(os.path.dirname(__file__), "ljpeg", "jpegdir", "jpeg")

if not os.path.exists(BIN):
    print "jpeg is not built yet; use 'cd jpegdir; make' first"
    sys.exit(0)

# sample output
#> GW:1979  GH:4349  R:0
#>> C:1  N:xx.ljpeg.1  W:1979  H:4349  hf:1  vf:1

PATTERN = re.compile('\sC:(\d+)\s+N:(\S+)\s+W:(\d+)\s+H:(\d+)\s')

def getLesionSize(mat):
    pos = [int(mat[1]), int(mat[0])]
    ext = [pos[0], pos[0], pos[1], pos[1]]

    translations = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

    for o in range(2,len(mat)):
        pos = numpy.subtract(pos, translations[int(mat[o])])
        ext[0] = min(ext[0], pos[0])
        ext[1] = max(ext[1], pos[0])
        ext[2] = min(ext[2], pos[1])
        ext[3] = max(ext[3], pos[1])

    return ext


def getOverlayData (path):
    path = path.replace('.LJPEG','.OVERLAY')
    f = open(path, 'r')
    boundary = f.read().replace('#','').replace('\n','').split('BOUNDARY')
    boundary = boundary[1].split('CORE')
    mat = filter(None, boundary[0].split(' '))
    return mat

def read (path):
    cmd = '%s -d -s %s' % (BIN, path)
    l = subprocess.check_output(cmd, shell=True)
    #print l
    m = re.search(PATTERN, l)
    C = int(m.group(1)) # I suppose this is # channels
    F = m.group(2)
    W = int(m.group(3))
    H = int(m.group(4))
    assert C == 1
    im = numpy.fromfile(F, dtype='uint16').reshape(H, W)
    L = im >> 8
    H = im & 0xFF
    im = (H << 8) | L
    os.remove(F)
    return im

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    import glob
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("ljpeg", nargs=1)
    parser.add_argument("output", nargs=1)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--scale", type=float)

    args = parser.parse_args()
    path = args.ljpeg[0]
    tiff = args.output[0]

    assert 'LJPEG' in path

    root = os.path.dirname(path)
    stem = os.path.splitext(path)[0]

    # read ICS
    ics = glob.glob(root + '/*.ics')[0]
    name = path.split('.')[-2]

    W = None
    H = None
    # find the shape of image
    for l in open(ics, 'r'):
        l = l.strip().split(' ')
        if len(l) < 7:
            continue
        if l[0] == name:
            W = int(l[4])
            H = int(l[2])
            bps = int(l[6])
            if bps != 12:
                logging.warn('BPS != 12: %s' % path)
            break

    assert W != None
    assert H != None

    image = read(path)

    if W != image.shape[1]:
        logging.warn('reshape: %s' % path)
        image = image.reshape((H, W))

    raw = image

    if args.visual:
        logging.warn("normalizing color, will lose information")
        if args.verify:
            logging.error("verification is going to fail")
        if args.scale:
            rows, cols = image.shape
            image = cv2.resize(image, (int(cols * args.scale), int(rows * args.scale)))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = numpy.uint8(image)
        mat = getOverlayData(path)
        ext = getLesionSize(mat)
        #cv2.imwrite('patch_'+tiff, image[ext[0]:ext[1], ext[2]:ext[3]])
        #image[ext[0]:ext[1], ext[2]:ext[3]] = 0
        #print(numpy.diff(ext))

    elif args.scale:
        logging.error("--scale must be used with --visual")
        sys.exit(1)
        #image = cv2.equalizeHist(image)
    #tiff = stem + '.TIFF'
    cv2.imwrite(tiff, image)

    if args.verify:
        verify = cv2.imread(tiff, -1)
        if numpy.all(raw == verify):
            logging.info('Verification successful, conversion is lossless')
        else:
            logging.error('Verification failed: %s' % path)

