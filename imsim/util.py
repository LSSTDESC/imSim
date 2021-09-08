import logging
import sys

# XXX: unneeded?
def get_logger(log_level, name=None):
    """
    Set up standard logging module and set lsst.log to the same log
    level.

    Parameters
    ----------
    log_level: str
        This is converted to logging.<log_level> and set in the logging
        config.
    name: str [None]
        The name to preprend to the log message to identify different
        logging contexts.  If None, then the root context is used.
    """
    # Setup logging output.
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(eval('logging.' + log_level))

    return logger


