from galsim.config.extra import ExtraOutputBuilder, RegisterExtraOutput
from galsim import PhotonArray
from astropy.io.fits import BinTableHDU

# An extra output type to enable two-pass photon scattering.
# This output should be used in the first pass to write scattered photons to file,
# while the second pass will read the scattered photons then go over all the images
# again and add the photons.
class ScatteredPhotonsBuilder(ExtraOutputBuilder):
    """Build pickled photon arrays containing photons which were not accumulated
    to the sensor during each image's construction.
    """

    def processImage(self, index, obj_nums, config, base, logger):
        """After each image is drawn, concatenate the list of scattered photon arrays
        and store in data ready to be pickled to file later on.
        """
        # If scattered photons have been stored in the base config, concatenate these
        # photon arrays to a single one and store in data, indexing by image number.
        if 'scattered_photons' in base and len(base['scattered_photons']) > 1:
            self.data[index] = PhotonArray.concatenate(base['scattered_photons'])
        else:
            self.data[index] = PhotonArray(N=0)

    def finalize(self, config, base, main_data, logger):
        return self.data
                
    def writeFile(self, file_name, config, base, logger):
        """Write the photon array to fits file.
        Might it be faster to write to a pickle file?
        """
        for photon_array in self.final_data:
            photon_array.write(file_name)

    def writeHdu(self, config, base, logger):
        """We don't want to write the scattered photons to FITS, so return an empty
        BinTable in lieu of something else.
        """
        return BinTableHDU(data=None)


RegisterExtraOutput('scattered_photons', ScatteredPhotonsBuilder())
