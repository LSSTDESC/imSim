"""
Code to test the performance of imsim for the sky background and drawing
objects.
"""
import os
import time
from collections import OrderedDict
import numpy as np
import galsim
from lsst.sims.photUtils import BandpassDict
from lsst.sims.GalSimInterface import SNRdocumentPSF
import desc.imsim

desc.imsim.read_config()
instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                            'tiny_instcat.txt')
instcat = desc.imsim.parsePhoSimInstanceFile(instcat_file)

class GalSimBandpasses(object):
    def __init__(self):
        self.lsst_bandpasses = BandpassDict.loadBandpassesFromFiles()[0]
        self._gs_bandpasses = dict()
    def __call__(self, bandpass_name):
        try:
            gs_bp = self._gs_bandpasses[bandpass_name]
        except KeyError:
            bp = self.lsst_bandpasses[bandpass_name]
            index = np.where(bp.sb != 0)
            gs_bp = galsim.Bandpass(galsim.LookupTable(x=bp.wavelen[index],
                                                       f=bp.sb[index]),
                                    wave_type='nm')
            self._gs_bandpasses[bandpass_name] = gs_bp
        return gs_bp

gs_bandpasses = GalSimBandpasses()

def sky_bg_timing(skymodel, nxy_min=10, nxy_max=300, npts=8, pixel_scale=0.2):
    timing = OrderedDict()
    nphot = OrderedDict()
    for nxy in [int(x) for x in np.logspace(np.log10(nxy_min),
                                            np.log10(nxy_max), npts)]:
        image = galsim.Image(nxy, nxy, scale=pixel_scale)
        t0 = time.clock()
        skymodel.addNoiseAndBackground(image, photParams=skymodel.photParams)
        timing[nxy] = time.clock() - t0
        try:
            nphot[nxy] = skymodel.photon_array.size()
        except AttributeError:
            nphot[nxy] = 0
    return timing, nphot

class StarTimer(object):
    def __init__(self, opsim_data, seed, nxy=64, pixel_scale=0.2):
        self.psf = SNRdocumentPSF(opsim_data.OpsimData['FWHMgeom'])
        self._rng = galsim.UniformDeviate(seed)
        self.nxy = nxy
        self.pixel_scale = pixel_scale

        fratio = 1.234
        obscuration = 0.606
        angles = galsim.FRatioAngles(fratio, obscuration, self._rng)

        self.bandpass = gs_bandpasses(opsim_data.bandpass)
        self.sed = galsim.SED(lambda x: 1, 'nm',
                              'flambda').withFlux(1., self.bandpass)
        waves = galsim.WavelengthSampler(sed=self.sed, bandpass=self.bandpass,
                                         rng=self._rng)
        self.surface_ops = (waves, angles)

    def get_sensor(self, nrecalc):
        return galsim.SiliconSensor(rng=self._rng, nrecalc=nrecalc)

    def timing(self, flux, sensor=None):
        point = galsim.DeltaFunction(flux=flux)
        star = galsim.Convolve(point*self.sed, self.psf._getPSF())

        image = galsim.Image(self.nxy, self.nxy)
        surface_ops = self.surface_ops if sensor is not None else ()
        t0 = time.clock()
        image = star.drawImage(method='phot', bandpass=self.bandpass,
                               image=image, scale=self.pixel_scale,
                               rng=self._rng, sensor=sensor,
                               surface_ops=surface_ops)
        return time.clock() - t0

    def flux_loop_timing(self, apply_model, nrecalc=10000, flux_min=10,
                         flux_max=1e5, nflux=8):
        sensor = self.get_sensor(nrecalc) if apply_model else None
        timing = OrderedDict()
        for flux in np.logspace(np.log10(flux_min), np.log10(flux_max), nflux):
            timing[flux] = self.timing(flux, sensor=sensor)
        return timing


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="imsim timing tests")
    parser.add_argument('--test_type', choices='sky_bg star'.split(),
                        default='sky_bg')
    parser.add_argument('--fast_sky_bg', action='store_true')
    parser.add_argument('--apply_sensor_model', action='store_true')
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()

    if args.test_type == 'sky_bg':
        skymodel = desc.imsim.make_sky_model(instcat.obs_metadata,
                                             instcat.phot_params,
                                             seed=args.seed,
                                             addNoise=False,
                                             addBackground=True,
                                             apply_sensor_model=args.apply_sensor_model)
        print(sky_bg_timing(skymodel))
    else:
        star_timer = StarTimer(opsim_data, args.seed)
        for nrecalc in (1000, 3e3):
            print(nrecalc,
                  star_timer.flux_loop_timing(args.apply_sensor_model,
                                              nrecalc=nrecalc))
