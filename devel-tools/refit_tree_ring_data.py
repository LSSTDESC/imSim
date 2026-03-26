import os
from itertools import pairwise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from lsst.obs.lsst import LsstCam
from imsim import TreeRings


tree_ring_file = os.path.join(
        os.environ["IMSIM_DIR"], "data",
        "tree_ring_data", "tree_ring_parameters_2018-04-26.txt"
    )
tree_rings = TreeRings(tree_ring_file)

camera = LsstCam.getCamera()

rmin = 3000.
rmax = 5500.

# HyeYun's tree ring centers refer to the lower left CCD corner as the
# xy origin, whereas Craig's tree ring code uses a nominal ccd as the
# xy origin with the following offsets.  These offsets need to be
# subracted from HyeYun's Cx, Cy values.
x_offset = 2048.5
y_offset = 2048.5
df = pd.read_csv("LSSTCam_data/HyeYun_tree_ring_center_2025.csv")
df['x0'] -= x_offset
df['y0'] -= y_offset

new_centers = dict(zip(df['ID'], zip(df['x0'], df['y0'])))

make_plots = False
if make_plots:
    outdir = "profile_plots"
    os.makedirs(outdir, exist_ok=True)
    plt.figure(1, figsize=(10, 4))

for detector, det in zip(range(189), camera):
    det_name = det.getName()
    tr_lut = pd.read_csv(f"LSSTCam_data/Tree_ring_LUT_{det_name}.csv")
    # Omit Nans and just consider radii in the range (rmin, rmax)
    df = tr_lut.query("amplitude==amplitude and "
                      f"radius > {rmin} and radius < {rmax}")

    # Bin by radius and fit the amplitude function to the stdev of each bin.
    radius = df['radius'].to_numpy()
    amplitude = df['amplitude'].to_numpy()

    nbins = 50  # (rmax - rmin)/nbins = 2500/50 = 50 pixels per bin
    indexes = np.linspace(0, len(df), nbins, dtype=int)
    x, y = [], []
    for imin, imax in pairwise(indexes):
        x.append(np.mean(radius[imin:imax]))
        y.append(np.std(amplitude[imin:imax]))

    xvals = np.array(x)
    yvals = np.array(y)

    XSCALE = 30000.0

    def amp(x, p, xscale=XSCALE):
        return p[0] + p[1]*(x/xscale)**4

    def residuals(p, x, y):
        return amp(x, p) - y

    bounds = ([0, 0], [1, 10])
    result = least_squares(residuals, [0, 1e-4], bounds=bounds,
                           args=(xvals, yvals))
    pfit = result.x
    pfit[1] /= XSCALE**4
    print(det_name, pfit)
    Cx, Cy = new_centers[det_name]
    if np.isnan(Cx):
        Cx = None
    if np.isnan(Cy):
        Cy = None
    tree_rings.update_info_block(
        det_name, Cx=Cx, Cy=Cy, A=pfit[0], B=2.0*pfit[1])

    if make_plots:
        plt.clf()
        plt.plot(df['radius'], df['amplitude'])
        plt.xlabel("radius (pixels)")
        plt.ylabel("tree ring amplitude")
        plt.title(det_name)
        plt.plot(xvals, yvals, color='red')
        plt.plot(xvals, amp(xvals, pfit, xscale=1), color='green')
        # Plot the updated tree ring function.
        func = tree_rings.get_func(det_name)
        plt.plot(radius, func(radius), color='orange')
        pngfile = os.path.join(outdir, f"{det_name}_tree_ring_profile.png")
        plt.savefig(pngfile)

outfile = "tree_ring_parameters_2026-03-25.txt"
tree_rings.write(outfile, overwrite=True)
