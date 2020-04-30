#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""
2015 December 10:
This is a modified version of the approximateWcs.py script from
meas_astrom
"""
import numpy as np
import lsst.afw.table as afwTable
import lsst.geom as LsstGeom
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.astrom.sip import makeCreateWcsWithSip

__all__ = ["approximateWcs"]

def approximateWcs(wcs, camera_wrapper=None, detector_name=None,
                   obs_metadata=None,
                   order=3, nx=20, ny=20, iterations=3,
                   skyTolerance=0.001*LsstGeom.arcseconds, pixelTolerance=0.02):
    """
    Approximate an existing WCS as a TAN-SIP WCS

    The fit is performed by evaluating the WCS at a uniform grid of
    points within a bounding box.

    @param[in] wcs  wcs to approximate
    @param[in] camera_wrapper is an instantiation of GalSimCameraWrapper
    @param[in] detector_name is the name of the detector
    @param[in] obs_metadata is an ObservationMetaData characterizing
    the telescope pointing
    @param[in] order  order of SIP fit
    @param[in] nx  number of grid points along x
    @param[in] ny  number of grid points along y
    @param[in] iterations number of times to iterate over fitting
    @param[in] skyTolerance maximum allowed difference in world
               coordinates between input wcs and approximate wcs
               (default is 0.001 arcsec)
    @param[in] pixelTolerance maximum allowed difference in pixel
               coordinates between input wcs and approximate wcs
               (default is 0.02 pixels) @return the fit TAN-SIP WCS
    """
    tanWcs = wcs

    # create a matchList consisting of a grid of points covering the bbox
    refSchema = afwTable.SimpleTable.makeMinimalSchema()
    refCoordKey = afwTable.CoordKey(refSchema["coord"])
    refCat = afwTable.SimpleCatalog(refSchema)

    sourceSchema = afwTable.SourceTable.makeMinimalSchema()
    SingleFrameMeasurementTask(schema=sourceSchema) # expand the schema
    sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])

    sourceCat = afwTable.SourceCatalog(sourceSchema)

    # 20 March 2017
    # the 'try' block is how it works in swig;
    # the 'except' block is how it works in pybind11
    try:
        matchList = afwTable.ReferenceMatchVector()
    except AttributeError:
        matchList = []

    bbox = camera_wrapper.getBBox(detector_name)
    bboxd = LsstGeom.Box2D(bbox)

    for x in np.linspace(bboxd.getMinX(), bboxd.getMaxX(), nx):
        for y in np.linspace(bboxd.getMinY(), bboxd.getMaxY(), ny):
            pixelPos = LsstGeom.Point2D(x, y)

            ra, dec = camera_wrapper.raDecFromPixelCoords(
                np.array([x]), np.array([y]),
                detector_name,
                obs_metadata=obs_metadata,
                epoch=2000.0,
                includeDistortion=True)

            skyCoord = LsstGeom.SpherePoint(ra[0], dec[0], LsstGeom.degrees)

            refObj = refCat.addNew()
            refObj.set(refCoordKey, skyCoord)

            source = sourceCat.addNew()
            source.set(sourceCentroidKey, pixelPos)

            matchList.append(afwTable.ReferenceMatch(refObj, source, 0.0))

    # The TAN-SIP fitter is fitting x and y separately, so we have to
    # iterate to make it converge
    for _ in range(iterations):
        sipObject = makeCreateWcsWithSip(matchList, tanWcs, order, bbox)
        tanWcs = sipObject.getNewWcs()
    fitWcs = sipObject.getNewWcs()

    return fitWcs
