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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--comcam", action="store_true", help="Setup for ComCam instead of LSSTCam")
args = parser.parse_args()


instcat_file = 'instcat_vignetting.txt'
opsim_data = imsim.OpsimDataLoader(instcat_file)

ra = opsim_data.get('fieldRA')
dec = opsim_data.get('fieldDec')
rottelpos = opsim_data.get('rotTelPos')
mjd = opsim_data.get('mjd')
band = opsim_data.get('band')

if args.comcam:
    nsamp = 300  # Fewer CCDs, so need more stars per CCD.
    camera_name = 'LsstComCamSim'
    center_ccd = 4
else:
    nsamp = 30
    camera_name = 'LsstCam'
    center_ccd = 94

camera = imsim.get_camera(camera_name)

wcs = imsim.make_batoid_wcs(ra, dec, rottelpos, mjd, band, camera_name)
fp_to_pixel = camera[center_ccd].getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
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
