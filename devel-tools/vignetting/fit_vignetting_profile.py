"""
Script to fit vignetting profile to sources rendered on the focal
plane in a collection of eimage files.
"""
import os
import glob
import json
from collections import defaultdict
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import lsst.afw.detection as afw_detect
import lsst.afw.fits as afw_fits
import lsst.afw.image as afw_image
import lsst.afw.math as afw_math
from lsst.afw import cameraGeom
import lsst.geom
from lsst.obs.lsst import LsstCam, LsstComCamSim
from argparse import ArgumentParser


def fp_signal(fp, image):
    """Return the flux contained within a footprint."""
    spans = fp.getSpans()
    total = 0
    for span in spans:
        y = span.getY()
        x0, x1 = span.getX0(), span.getX1() + 1
        total += sum(image.array[y][x0:x1])
    return total

class Transforms:
    """
    Class to manage access to cameraGeom coordinate transfomation
    objects that go from pixel coordinates on an individual CCD to
    focal plane coordinates in mm.
    """
    def __init__(self, camera):
        self.camera = camera
        self.data = {}
    def __getitem__(self, det_name):
        if det_name not in self.data:
            self.data[det_name] \
                = self.camera[det_name].getTransform(cameraGeom.PIXELS,
                                                     cameraGeom.FOCAL_PLANE)
        return self.data[det_name]

def extract_fluxes(eimage_file, thresh=10, grow=1):
    """
    Use lsst.afw.detection to find sources on a background-subtracted
    image, compute the source fluxes by integrating over the detection
    footprints, and set the source location to the position of the
    first footprint peak.

    Return a pandas DataFrame with this info for each detected source.
    """
    data = defaultdict(list)
    md = afw_fits.readMetadata(eimage_file)
    det_name = md.get('DET_NAME')
    image = afw_image.ImageD(item)
    threshold = afw_detect.Threshold(thresh)
    fp_set = afw_detect.FootprintSet(image, threshold)
    fp_set = afw_detect.FootprintSet(fp_set, grow, False)
    for fp in fp_set.getFootprints():
        data['det_name'].append(det_name)
        data['flux'].append(fp_signal(fp, image))
        peak = list(fp.getPeaks())[0]
        centroid = peak.getCentroid()
        data['x'].append(centroid.x)
        data['y'].append(centroid.y)
    return pd.DataFrame(data)

parser = ArgumentParser()
parser.add_argument("--comcam", action="store_true", help="Setup for ComCam instead of LSSTCam")
args = parser.parse_args()

if args.comcam:
    camera = LsstComCamSim.getCamera()
    eimage_files = sorted(glob.glob('output_vignetting_ComCam/eimage*.fits'))
    data_file = 'vignetting_data_comcam.parq'
else:
    camera = LsstCam.getCamera()
    eimage_files = sorted(glob.glob('output_vignetting_LsstCam/eimage*.fits'))
    data_file = 'vignetting_data_lsstcam.parq'
print(len(eimage_files))

if not os.path.isfile(data_file):
    # Extract source fluxes from each CCD and aggregate into a data frame.
    dfs = []
    for item in tqdm(eimage_files):
        dfs.append(extract_fluxes(item, thresh=10, grow=1))
    df0 = pd.concat(dfs)

    # Loop over sources and convert CCD pixel coordinates to focal
    # plane coordinates.
    transforms = Transforms(camera)
    fp_x, fp_y = [], []
    for _, row in df0.iterrows():
        transform = transforms[row.det_name]
        loc = transform.applyForward(lsst.geom.Point2D(int(row.x), int(row.y)))
        fp_x.append(loc[0])
        fp_y.append(loc[1])
    df0['fp_x'] = fp_x
    df0['fp_y'] = fp_y
    df0['radius'] = np.sqrt(df0['fp_x']**2 + df0['fp_y']**2)

    # Write the data to a parquet file.
    df0.to_parquet(data_file)
else:
    # Avoid recomputing if the file exists already.
    df0 = pd.read_parquet(data_file)

plt.figure(figsize=(8, 4))

# Plot star locations on focal plane
nstars = len(df0) // len(eimage_files)
plt.subplot(1, 2, 1)
plt.scatter(df0['fp_x'], df0['fp_y'], s=2)
plt.gca().set_aspect('equal')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title(f'Star positions for vignetting evaluation,\n{nstars} per CCD, mag_i = 22.5', fontsize='small')

# Fit a spline to the profile.
#
# Exclude out-of-family values that may be blends or objects too close
# to the CCD edge.
df = df0.query('1000 < flux < 2.0e5')
x = df['radius'].to_numpy()
y = df['flux'].to_numpy()
index = np.argsort(x)  # Spline fitting code needs points sorted in x.
x = x[index]
y = y[index]

knots = 6
x_new = np.linspace(0, 1, knots + 2)[1:-1]
q_knots = np.quantile(x, x_new)

# Add locations of sizeable bends in the profile.
if args.comcam:
    q_knots = sorted(list(q_knots) + [62., 65.])
else:
    q_knots = sorted(list(q_knots) + [312.5, 332.5, 347])

# Get continuity at the origin by reflecting the input data + knots
x = np.concatenate((-x[::-1], x))
y = np.concatenate((y[::-1], y))
q_knots = np.concatenate(([-q for q in reversed(q_knots)], q_knots))

# plot the vignetting profile data and spline fit
plt.subplot(1, 2, 2)
plt.scatter(x, y, s=2, label='stars')

# Retrieve the knot locations (t), spline coefficients (c), and spline
# order (k)
tck = interpolate.splrep(x, y, t=q_knots, s=1)
xx = np.linspace(0, max(x)+5, 100)
yfit = interpolate.BSpline(*tck)(xx)
plt.plot(xx, yfit, color='red', label='spline fit')

plt.legend(fontsize='x-small')
plt.xlabel('distance from focal plane center (mm)')
plt.ylabel('flux (ADU)')
plt.title(f'Radial profile of vignetting for {camera.getName()} simulation',
          fontsize='small')
plt.xlim(0, plt.xlim()[1])

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(f'{camera.getName()}_vignetting_profile_fit.png', dpi=300)

# Save the data as a json file, converting np.arrays to serializable
# lists.
with open(f'{camera.getName()}_vignetting_spline.json', 'w') as fobj:
    json.dump((list(tck[0]), list(tck[1]), tck[2]), fobj)
