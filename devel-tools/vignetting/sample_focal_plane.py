"""
Script to create instcat star entries that sample the LSSTCam focal plane.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import galsim
from lsst.afw import cameraGeom
import lsst.geom
import imsim

# Pointing info for DC2 visit 182850:
ra = 51.99085849573259566
dec = -40.31737846575015283
rottelpos = 28.8262515
mjd = 59822.28563761110854102
band = 'i'

# The number of stars per CCD.
nsamp = 10

camera_name = 'LsstCam'
camera = imsim.get_camera(camera_name)

wcs = imsim.make_batoid_wcs(ra, dec, rottelpos, mjd, band, camera_name)
fp_to_pixel = camera[94].getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
data = defaultdict(list)
for i, det in enumerate(camera):
    if det.getType() != cameraGeom.DetectorType.SCIENCE:
        # Only consider science CCDs.
        continue
    pixel_to_fp = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    xvals = np.random.uniform(50, 3950, size=nsamp)  # avoid CCD edges
    yvals = np.random.uniform(50, 3950, size=nsamp)
    for x, y in zip(xvals, yvals):
        loc = pixel_to_fp.applyForward(lsst.geom.Point2D(x, y))
        pixels = fp_to_pixel.applyForward(loc)
        image_pos = galsim.PositionD(pixels.x, pixels.y)
        skyCoord = wcs.toWorld(image_pos)
        data['ra'].append(skyCoord.ra/galsim.degrees)
        data['dec'].append(skyCoord.dec/galsim.degrees)
        data['x'].append(x)
        data['y'].append(y)
        data['det_name'].append(det.getName())
df0 = pd.DataFrame(data)
#df0.to_parquet('fp_sample.parq')

magnorm = 22.25
template = "object %(object_id)i %(ra).10f %(dec).10f %(magnorm).3f starSED/phoSimMLT/lte027-2.0-0.0a+0.0.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.04507462 3.1\n"

instcat_file = 'stars_vignetting.txt'
with open(instcat_file, 'w') as fobj:
    for object_id, (_, row) in enumerate(df0.iterrows()):
        ra = row.ra
        dec = row.dec
        fobj.write(template % locals())
