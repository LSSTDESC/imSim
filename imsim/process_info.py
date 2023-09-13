import os
import time
from collections import namedtuple
import psutil
import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput


_PROCESS_INFO_COLS = ['object_id', 'pid', 'rss', 'uss', 'user_time',
                      'unix_time']
_PROCESS_INFO_TYPES = [str, int, float, float, float, float]
ProcessInfo = namedtuple('ProcessInfo', _PROCESS_INFO_COLS)


class ProcessInfoBuilder(ExtraOutputBuilder):
    """
    Build output file with per-process cpu and memory info extracted
    using psutil.
    """
    def processStamp(self, obj_num, config, base, logger):
        object_id = base.get('object_id')

        pid = os.getpid()
        proc = psutil.Process(pid)
        mem_full_info = proc.memory_full_info()

        # Compute memory in GiB
        rss = mem_full_info.rss/1024**3
        uss = mem_full_info.uss/1024**3

        # User cpu time
        user_time = proc.cpu_times().user

        process_info = ProcessInfo(object_id, pid, rss, uss, user_time,
                                   time.time())
        logger.info("Object %d, id %s, pid %d, RSS %.2f GB, USS %.2f GB, "
                    "user_time %.2f, unix_time %.1f, det_name %s",
                    obj_num, *process_info, base.get("det_name", ""))
        self.scratch[obj_num] = process_info

    def finalize(self, config, base, main_data, logger):
        self.cat = galsim.OutputCatalog(names=_PROCESS_INFO_COLS,
                                        types=_PROCESS_INFO_TYPES)
        obj_nums = sorted(self.scratch.keys())
        for obj_num in obj_nums:
            self.cat.addRow(self.scratch[obj_num])
        return self.cat

    def writeFile(self, file_name, config, base, logger):
        self.cat.write(file_name)


RegisterExtraOutput('process_info', ProcessInfoBuilder())
