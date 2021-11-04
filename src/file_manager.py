# File Manager code

import os
import glob

class FMan(object):

    dataPath = '../Data/'
    imageVersion = 1

    def __init__(self):
        pass

    def importNetwork(self):
        pass

    def exportNetwork(self):
        pass

    def preProcessImages(self):
        toprocess = []
        imagelist = glob.glob(os.path.join(self.dataPath, '*.png'))
        for ipath in imagelist:
            if(os.path.exists(ipath.replace('.png','.bmp'))):
                nf = open(ipath.replace('.png','.bmp'))
                ver = int(nf.read())
                if ver != self.imageVersion:
                    os.remove(ipath.replace('.png','.bmp'))
                    toprocess.append(ipath)
            else:
                toprocess.append(ipath)

        for image in imagelist:
            pass