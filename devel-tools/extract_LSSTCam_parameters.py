import os
from collections import defaultdict
import pickle
import numpy as np
import lsst.afw.image as afwImage
import lsst.daf.butler as daf_butler
from lsst.ip.isr import Defects
from lsst.obs.lsst import LsstCam


def get_amp_data(camera, repo, collections):
    """
    Retrieve per-amp values for gain, read noise, and ptc turnoff from
    the ptc datasets from the specified collections.
    """
    butler = daf_butler.Butler(repo, collections=collections)
    refs = set(butler.registry.queryDatasets('ptc', findFirst=True))

    ptc_attributes = {'gain': 'gain',
                      'saturation': 'ptcTurnoff',
                      'read_noise': 'noise'}

    amp_data = defaultdict(lambda: defaultdict(dict))
    for column, attr in ptc_attributes.items():
        for ref in refs:
            ptc = butler.get(ref)
            det = camera[ref.dataId['detector']]
            det_name = det.getName()
            for amp in det:
                amp_name = amp.getName()
                amp_data[column][det_name][amp_name] \
                    = ptc.__dict__[attr][amp_name]

    amp_data = {key: dict(value) for key, value in amp_data.items()}

    return amp_data


def merge_defects(detector, defects_list):
    """
    Merge defects from a list of cpPartialDefects, which are extracted
    from individual exposures.
    """
    if not defects_list:
        return None
    sumImage = afwImage.MaskedImageF(detector.getBBox())
    count = 0
    for defects in defects_list:
        count += 1
        for defect in defects:
            sumImage.image[defect.getBBox()] += 1.0
    sumImage /= count
    nDetected = len(np.where(sumImage.getImage().getArray() > 0)[0])
    print(detector.getName(), end=": ")
    print(f"Pre-merge {nDetected} pixels with non-zero detections.")
    threshold = 0.7   # Fraction of exposures required to include a defect.
    indices = np.where(sumImage.getImage().getArray() > threshold)
    BADBIT = sumImage.getMask().getPlaneBitMask('BAD')
    sumImage.getMask().getArray()[indices] |= BADBIT
    print(f"Post-merge {len(indices[0])} pixels marked as defects.\n")
    merged = Defects.fromMask(sumImage, 'BAD')
    return merged


def get_defects(camera, repo, collections, acq_run):
    """
    Get the merged bright and dark defects for all of the detectors
    measured from the specified acquisition run.
    """
    butler = daf_butler.Butler(repo, collections=collections)

    defects = defaultdict(dict)
    for image_type, defect_type in (('dark', 'bright'), ('flat', 'dark')):
        where = (f"exposure.science_program='{acq_run}' "
                 f"and exposure.observation_type='{image_type}'")
        ref_set = set(butler.registry.queryDatasets('cpPartialDefects',
                                                    where=where).expanded())
        refs = defaultdict(list)
        for ref in ref_set:
            refs[ref.dataId['detector']].append(ref)

        for det in camera:
            detector = det.getId()
            det_name = det.getName()
            defects_list = [butler.get(ref) for ref in refs[detector]]
            defects[defect_type][det_name] = merge_defects(det, defects_list)

    return defects


def save_defects(defects, output_dir):
    """
    Save the per-CCD defects as FITS files in a directory tree.
    """
    for defect_type, defects_dict in defects.items():
        outdir = os.path.join(output_dir, defect_type)
        os.makedirs(outdir, exist_ok=True)
        for det_name, defect_data in defects_dict.items():
            outfile = os.path.join(outdir,
                                   f"{det_name}_{defect_type}_defects.fits")
            defect_data.writeFits(outfile)


if __name__ == '__main__':
    camera = LsstCam.getCamera()
    repo = '/repo/ir2'

#    collections = ['u/lsstccs/ptc_13412_w_2023_25']
#    amp_data = get_amp_data(camera, repo, collections)
#    with open('ptc_data_13412.pickle', 'wb') as fobj:
#        pickle.dump(amp_data, fobj)

#    collections = ['u/lsstccs/defects_13391_w_2023_24']
#    defects = get_defects(camera, repo, collections, 13391)
#    save_defects(defects, 'defects_13391')
