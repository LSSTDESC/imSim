import numpy as np
from lsst.obs.lsst import LsstCam
from lsst.afw.cameraGeom import FOCAL_PLANE
import imsim
import matplotlib.pyplot as plt

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

camera = LsstCam().getCamera()

det_telescope = imsim.DetectorTelescope(
    "LSST_r.yaml",
    fp_height_file_name="fp_height_map.ecsv",
)

x = []
y = []
z = []
for det in camera:
    corners = det.getCorners(FOCAL_PLANE)
    xmin = min(corner.getX() for corner in corners)
    xmax = max(corner.getX() for corner in corners)
    ymin = min(corner.getY() for corner in corners)
    ymax = max(corner.getY() for corner in corners)

    # Transpose DVCS => CCS
    xmin, ymin = ymin, xmin
    xmax, ymax = ymax, xmax

    xs = np.linspace(xmin, xmax, 10)
    ys = np.linspace(ymin, ymax, 10)
    xs, ys = np.meshgrid(xs, ys)

    # Build the det_telescope
    z_offset = det_telescope.calculate_z_offset(det.getName())
    height_map = det_telescope.calculate_height_map(det.getName())
    telescope = det_telescope.get_telescope(z_offset, height_map)
    zs = telescope['Detector'].surface.sag(xs*1e-3, ys*1e-3)

    x.append(xs)
    y.append(ys)
    z.append(zs)

x = np.array(x)  # Already in mm
y = np.array(y)
z = np.array(z)*1e6  # Convert to micron

# Remove WF sensor 1.5 mm offsets for plot
z[z > 1000] -= 1500
z[z < -1000] += 1500

# vmax = np.quantile(z, 0.95)
vmax = 15.0

# To compare with Document-37242, we plot X_CCS on the Y axis,
# Y_CCS on the X axis, and heights in -Z_CCS.
plt.figure(figsize=(10, 10))
colorbar(plt.scatter(y, x, c=-z, vmin=-vmax, vmax=vmax, cmap='jet', s=5))
plt.title("Height Map (micron)")
plt.xlabel("Y_CCS (mm)")
plt.ylabel("X_CCS (mm)")
plt.savefig("fp_height_map.png", dpi=300)

