"""
Module to record the RSS and USS memory usage of subprocesses that use
the multiprocessing module.
"""
import os
import sys
import pwd
import time
try:
    import cPickle as pickle
except:
    import pickle
import subprocess
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import psutil

__all__ = ['process_monitor', 'RssHistory', 'plot_rss_history' ]

class RssHistory:
    """
    Class to contain the time history of the RSS and USS memory for a
    process.
    """
    def __init__(self):
        self.time = []
        self.rss = []
        self.uss = []
    def append(self, time, memory_full_info):
        """Append a time/RSS entry."""
        self.time.append(time)
        self.rss.append(memory_full_info.rss/1024.**3)
        self.uss.append(memory_full_info.uss/1024.**3)

def process_monitor(process_name=None, outfile=None, wait_time=30):
    """
    This function should be run as one of the subprocesses in the
    the mulitprocessing pool.

    Parameters
    ----------
    process_name: str [None]
        String to grep from ps auxww output for finding the desired
        subprocesses to monitor.  If None, then use the script name
        given in sys.argv[0].
    outfile: str [None]
        Filename of output pickle file to contain the RSS time history
        data.  If None, then build the name from the process id of
        this function, 'imsim_rss_info_<pid>.pkl'.
    wait_time: float [30]
        Sampling time in seconds.
    """
    if process_name is None:
        process_name = os.path.basename(sys.argv[0])

    my_pid = os.getpid()
    if outfile is None:
        outfile = 'imsim_rss_info_{}.pkl'.format(my_pid)

    process_memories = defaultdict(RssHistory)
    userid = pwd.getpwuid(os.getuid()).pw_name
    command = ' | '.join(("ps auxww", "grep {}".format(userid),
                          "grep {}".format(process_name), "grep -v grep"))

    while True:
        try:
            lines = subprocess.check_output(command, shell=True).decode('utf-8')
        except subprocess.CalledProcessError:
            break

        lines = lines.strip().split('\n')
        pids = sorted([int(line.split()[1]) for line in lines])

        for pid in pids:
            if pid == my_pid:  # Skip the process for this function.
                continue
            proc = psutil.Process(pid)
            process_memories[pid].append(time.time(), proc.memory_full_info())
        time.sleep(wait_time)
        with open(outfile, 'wb') as output:
            pickle.dump(process_memories, output)

def plot_rss_history(rss_info_file, uss=True):
    """
    Plot the RSS or USS history from the specified file.

    Parameters
    ----------
    rss_info_file: str
        The pickle file with the RSS vs time data for the monitored
        processes.
    uss: bool [True]
        Plot the unique set size memory.

    Return
    ------
    matplotlib.container.ErrorbarContainer:  The figure with the plot.
    """
    with open(rss_info_file, 'rb') as input_:
        data = pickle.load(input_)
    t0 = None
    for pid in data:
        if t0 is None:
            t0 = data[pid].time[0]
        ydata = data[pid].uss if uss else data[pid].rss
        fig = plt.errorbar((np.array(data[pid].time) - t0)/60., ydata,
                           fmt='-', label=str(pid))
    plt.xlabel('relative time (min)')
    ylabel = 'USS memory (GB)' if uss else 'RSS memory (GB)'
    plt.ylabel(ylabel)
    plt.legend(loc=0, fontsize='x-small')
    return fig, data

if __name__ == '__main__':
    import multiprocessing

    def count(nmax=120, wait=1):
        for _ in range(nmax):
            time.sleep(wait)

    pool = multiprocessing.Pool(processes=3)
    results = []
    results.append(pool.apply_async(process_monitor, ()))
    results.append(pool.apply_async(count, (120,)))
    results.append(pool.apply_async(count, (120,)))
    pool.close()
    pool.join()
    for res in results:
        res.get()
