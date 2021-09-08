
import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput

class CameraReadout(ExtraOutputBuilder):
    """This is a GalSim "extra output" builder to write out the amplifier file simulating
    the camera readout of the main "e-image".
    """

    def finalize(self, config, base, main_data, logger):
        """Perform any final processing at the end of all the image processing.

        This function will be called after all images have been built.

        It returns some sort of final version of the object.  In the base class, it just returns
        self.data, but depending on the meaning of the output object, something else might be
        more appropriate.

        Parameters:
           config:     The configuration field for this output object.
           base:       The base configuration dict.
           main_data:  The main file data in case it is needed.
           logger:     If given, a logger object to log progress. [default: None]

        Returns:
           The final version of the object.
        """
        logger.warning("Making amplifier images")
        # Simple version of this for now.
        # main_data is a list of the single e-image.
        # Split it up into 16 amplifier images.
        det_name = base['det_name']  # This was saved by the LSST CCD Builder.
        eimage = main_data[0]

        # TODO: Do something with these numbers.
        readout_time = config['readout_time']   # Maybe give default values for these using get?
        dark_current = config['dark_current']   # Or if you want them to be parsable float values,
        bias_level = config['bias_level']       # we can use the config ParseValue function.
        pcti = config['pcti']
        scti = config['scti']

        # I don't know which way is the short side, and I know some of these need to be
        # flipped in various ways.  Ignoring all that for now.
        # TODO: Do this right.
        xvals = [i * eimage.xmax / 8 for i in range(9)]
        yvals = [j * eimage.xmax / 2 for j in range(3)]
        bounds_list = [galsim.BoundsI(xvals[i]+1,xvals[i+1],yvals[j]+1,yvals[j+1])
                       for i in range(8) for j in range(2)]

        amps = [eimage[b] for b in bounds_list]

        for amp_num, amp in enumerate(amps):
            amp_name = det_name + "-C%02d"%amp_num  # XXX: Pretty sure this isn't right.
            logger.warning("Amp %s has bounds %s.",amp_name,amp.bounds)

            amp.header = galsim.FitsHeader()
            amp.header['AMP_NAME'] = amp_name
            # TODO: Add other header information.

            # XXX: I think these probably all should have their origin set to 1,1?
            #      At this point, the amp images still ahve the bounds from the original
            #      eimage, so we would need this next line to get back to 1,1 as the origin.
            amp.setOrigin(1,1)

            # TODO: I think probably also the WCS might need to be changed?  Not sure.
            #       At this point, the WCS should be accurate for the re-origined image.
            #       But if we do something more complicated like mirror image the array or
            #       something like that, the WCS would probably need to change.

        return amps

    def writeFile(self, file_name, config, base, logger):
        """Write this output object to a file.

        The base class implementation is appropriate for the cas that the result of finalize
        is a list of images to be written to a FITS file.

        Parameters:
            file_name:  The file to write to.
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]
        """
        logger.warning("Writing amplifier images to %s",file_name)
        # self.final_data is the output of finalize, which is our list of amp images.
        galsim.fits.writeMulti(self.final_data, file_name)


RegisterExtraOutput('readout', CameraReadout())
