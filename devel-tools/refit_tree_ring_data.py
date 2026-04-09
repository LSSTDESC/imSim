import os
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lsst.obs.lsst import LsstCam
from imsim import TreeRings


def boxcar(data, window_size, mode="same"):
    window = np.ones(window_size)/window_size
    return np.convolve(data, window, mode=mode)


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

make_plots = True
if make_plots:
    plt.ion()
    outdir = "profile_plots"
    os.makedirs(outdir, exist_ok=True)
    plt.figure(1, figsize=(10, 4))

for detector, det in zip(range(189), camera):
    det_name = det.getName()
    tr_lut = pd.read_csv(f"LSSTCam_data/Tree_ring_LUT_{det_name}.csv")
    # Omit Nans and just consider radii in the range (rmin, rmax)
    df = tr_lut.query("amplitude==amplitude and "
                      f"radius > {rmin} and radius < {rmax}")

    # Box car average to smooth out noise.
    window = 10
    radius = df['radius'].to_numpy()
    amplitude = boxcar(df['amplitude'].to_numpy(), window)

    # Scale model parameters by ratio of stdevs for LSSTCam data vs
    # current model.
    rmin_std = 4500.0
    rmax_std = 5200.0
    index = np.where((rmin_std < radius) & (radius < rmax_std))
    data_std = np.std(amplitude[index])

    model_func = tree_rings.get_dfdr(det_name)
    model_std = np.std(model_func(radius[index]))

    scale_factor = data_std/model_std
    print(det_name, scale_factor)

    A, B = [float(_) for _ in
            tree_rings.info_blocks[det_name][1].strip().split()[-2:]]
    A *= scale_factor
    B *= scale_factor

    Cx, Cy = new_centers[det_name]
    if np.isnan(Cx):
        Cx = None
    if np.isnan(Cy):
        Cy = None
    tree_rings.update_info_block(det_name, Cx=Cx, Cy=Cy, A=A, B=B)

    if make_plots:
        plt.clf()
        plt.plot(radius, amplitude - np.mean(amplitude))
        plt.xlabel("radius (pixels)")
        plt.ylabel("tree ring amplitude")
        plt.title(det_name)
        # Plot the updated tree ring function.
        dfdr = tree_rings.get_dfdr(det_name)
        plt.plot(radius, dfdr(radius), color='orange',
                 label=f"scale factor: {scale_factor:.2e}")
        plt.legend(fontsize='x-small')
        plt.axvline(rmin_std, linestyle=':')
        plt.axvline(rmax_std, linestyle=':')
        pngfile = os.path.join(outdir, f"{det_name}_tree_ring_profile.png")
        plt.savefig(pngfile)
#        plt.show()
#        if not click.confirm("continue?", default=True):
#            break

outfile = "tree_ring_parameters_2026-04-02.txt"
tree_rings.write(outfile, overwrite=True)
