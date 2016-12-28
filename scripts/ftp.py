from ftplib import FTP_TLS, error_perm
import os
import subprocess

def connect(url):
    ftp = FTP_TLS(url)
    print('Attempting to connect to %s...' % url)
    #print(ftp.login())
    ftp.sendcmd('USER anonymous')
    ftp.sendcmd('PASS ')
    ftp.retrlines('LIST')
    return ftp

# from http://stackoverflow.com/questions/8676508/recursively-list-ftp-directories-in-python
def ftp_walk(ftp):
    print('Path: %s' % ftp.pwd())
    dirs = ftp.nlst()
    if found_ics(ftp, dirs):
        return
    for item in (path for path in dirs if path not in ('.', '..')):
        try:
            ftp.cwd(item)
            print 'Changed to', ftp.pwd()
            try:
                ftp_walk(ftp)
            finally:
                ftp.cwd('..')
        except error_perm, e:
            print item, e

def found_ics(ftp, dirs):
    if any(".ics" in s for s in dirs):
        path = ftp.pwd().replace('/pub/',"")
        print "Processing", path
        if not os.path.exists(path):
            os.makedirs(path)
        for filename in (p for p in dirs if p.find('16_PGM') == -1):
            file = open(os.path.join(path, filename), 'wb')
            counter = Integer();
            def data_middleman(file, data, counter, size):
                counter += len(data)
                update_progress(counter._val*100/size)
                file.write(data)
            size = ftp.size(filename)
            print "downloading %s..." % filename
            ftp.retrbinary('RETR %s' % filename, lambda data: data_middleman(file, data, counter, size))
            file.close()
            print " "
            if not filename.find(".LJPEG") == -1:
                output = filename.replace('.LJPEG','.png')
                #print " ".join(["python",os.path.abspath("./ROIPatches_python/ljpeg.py"), os.path.join(path, filename), os.path.join(path, output), "--visual","--scale","0.5"])
                with open(os.devnull, 'w') as fp:
                    subprocess.Popen(["python", os.path.abspath("./ROIPatches_python/ljpeg.py"), os.path.join(path, filename), os.path.join(path, output), "--visual","--scale","0.25"], stdout=fp, stderr=fp)
        return True
    return False

class Integer(object) :
    def __init__(self, val=0) :
            self._val = int(val)
    def __add__(self, val) :
        self._val += val
        return self
    def __str__(self) :
        return str(self._val)
    def __repr__(self) :
        return 'Integer(%s)' %self._val

def update_progress(progress):
    print '\r[{0}] {1}%'.format('#'*(progress/10), progress),

def main():
    #ftp = connect('ftp.debian.org')
    #ftp = connect('figment.csee.usf.edu')
    ftp.cwd('pub/DDSM/cases/cancers')
    ftp_walk(ftp)
    #with open(os.devnull, 'w') as fp:
    #    subprocess.Popen("python /Users/AJ/Desktop/CNN-Mammography-classification/scripts/ROIPatches_python/ljpeg.py DDSM/cases/normals/normal_10/case3660/B_3660_1.LEFT_CC.LJPEG DDSM/cases/normals/normal_10/case3660/B_3660_1.LEFT_CC.png --visual".split(), stdout=fp, stderr=fp)

if __name__ == "__main__":
    main()