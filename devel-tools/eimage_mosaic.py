"""
Script to create a focal plane mosaic from eimage files.
"""
import os
import glob
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.afw.cameraGeom import utils as cgu
from lsst.obs.lsst import LsstCam


class EimageMap:
    def __init__(self, glob_pattern):
        self._map = {}
        eimage_files = sorted(glob.glob(glob_pattern))
        for item in eimage_files:
            det_name = os.path.basename(item).split('-')[-2]
            self._map[det_name] = item

    def __getitem__(self, key):
        return afwImage.ImageF(self._map[key])

    def keys(self):
        return self._map.keys()


class EimageSource:
    isTrimmed = True
    background = 0.0

    def __init__(self, eimage_map):
        self.eimage_map = eimage_map

    def getCcdImage(self, det, imageFactory, binSize=1, *args, **kwargs):
        ccdImage = self.eimage_map[det.getName()]
        if binSize != 1:
            ccdImage = afwMath.binImage(ccdImage, binSize)
        n_rot = det.getOrientation().getNQuarter()
        return afwMath.rotateImageBy90(ccdImage, n_rot), det

if __name__ == '__main__':
    camera = LsstCam.getCamera()
    eimage_map = EimageMap('output_instcat/eimage*')
    image_source = EimageSource(eimage_map)
    binSize = 4
    mosaic = cgu.showCamera(camera, imageSource=image_source,
                            detectorNameList=eimage_map.keys(),
                            binSize=binSize)
