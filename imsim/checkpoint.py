import os
import shutil
import h5py
import galsim
import pickle
import numpy as np
from galsim.config import RegisterInputType, InputLoader


class Checkpointer:
    """
    A helper class that handles all the I/O associated with checkpointing.

    Parameters:

        file_name:      The name of the file to write/read.
        logger:         A logger object to log progress.
    """
    _req_params = {'file_name': str}
    _opt_params = {'dir': str}

    def __init__(self, file_name, dir=None, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)

        self.file_name = file_name
        if dir is not None:
            self.file_name = os.path.join(dir, self.file_name)
        galsim.utilities.ensure_dir(self.file_name)

        # Guard against failures during I/O.
        # The mode we'll use for writing is the following:
        #   1. Move the existing file to the backup name (_bak)
        #   2. Copy the backup to a new name (_new)
        #   3. Edit the new file with any new information being checkpointed.
        #   4. Move the new file to the regular name.
        #   5. Delete the backup.

        self.file_name_bak = self.file_name + "_bak"
        self.file_name_new = self.file_name + "_new"

        # At the start, there are several possible states things could be in:
        #   A. No files written.  Starting from scratch.
        #   B. Failed between steps 1 and 4.  Recover from the backup.  Maybe delete new.
        #   C. Failed between steps 4 and 5.  Delete the backup.
        #   D. Successfully completed step 5.  Normal checkpointing case.

        # At this point, covert cases B or C into D.
        if os.path.isfile(self.file_name):
            # Cases C or D
            self.logger.warning("Checkpoint file %s exists.", self.file_name)
            if os.path.isfile(self.file_name_bak):
                # Case C
                self.logger.warning("Backup file %s also exists. Deleting.", self.file_name_bak)
                os.remove(self.file_name_bak)
        elif os.path.isfile(self.file_name_bak):
            # Case B
            self.logger.warning("Backup checkpoint file %s exists. Recovering.", self.file_name_bak)
            os.rename(self.file_name_bak, self.file_name)
            if os.path.isfile(self.file_name_new):
                self.logger.info("Also found file %s. Deleting.", self.file_name_new)
                os.remove(self.file_name_new)
        else:
            # Case A
            self.logger.info("No checkpoint file %s detected.", self.file_name)

    def save(self, name, data):
        """Save some data to the checkpoint file under the give name.
        """
        self.logger.debug('checkpointing: %s',name)
        # 1. First backup the current file before we edit anything.
        if os.path.isfile(self.file_name):
            os.rename(self.file_name, self.file_name_bak)
            # 2. Copy the file to a new name.
            shutil.copy(self.file_name_bak, self.file_name_new)
            self.logger.debug('copied existing file to %s',self.file_name_bak)

        # 3. Update the "name" entry in the new file.
        with h5py.File(self.file_name_new, 'a') as hdf:
            self.logger.debug('opened new file %s',self.file_name_new)
            if name in hdf:
                del hdf[name]

            data_str = pickle.dumps(data)
            # hdf5 doesn't like strings with NULL bytes, which this tends to have,
            # so pretend its uint8 data.
            arr = np.frombuffer(data_str, dtype=np.uint8)
            hdf.create_dataset(name, data=arr)
            self.logger.debug('wrote data')

        # 4. Move the new file back to the regular name.
        os.rename(self.file_name_new, self.file_name)
        self.logger.debug('moved new file to %s',self.file_name)

        # 5. Delete the backup file if we made one
        if os.path.isfile(self.file_name_bak):
            os.remove(self.file_name_bak)
            self.logger.debug('removed backup file %s',self.file_name_bak)
        self.logger.info('Finished checkpointing %s.  Size=%d', name, arr.size)

    def load(self, name):
        """Load the existing data in the checkpoint file with the given name.
        Or return None if no checkpoint file exists.

        Trying to read an invalid name from an existing checkpoint file will raise an exception.
        """
        if not os.path.isfile(self.file_name):
            self.logger.debug('no checkpoint file yet')
            return None

        self.logger.debug('loading %s',name)
        with h5py.File(self.file_name, 'r') as hdf:
            if name not in hdf:
                self.logger.debug('nothing checkpointed for %s',name)
                return None
            arr = hdf[name][:]
            data_str = arr.tobytes()
            data = pickle.loads(data_str)
            self.logger.info('Loaded checkpointed data for %s.  Size=%d', name, arr.size)

        return data

RegisterInputType('checkpoint', InputLoader(Checkpointer, takes_logger=True))
