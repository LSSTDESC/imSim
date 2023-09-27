from contextlib import contextmanager
import logging

# This file has any helper functions/classes that are used by tests from multiple files.

# Copied verbatim from GalSim:
# https://github.com/GalSim-developers/GalSim/blob/releases/2.4/tests/galsim_test_helpers.py
class CaptureLog:
    """A context manager that saves logging output into a string that is accessible for
    checking in unit tests.

    After exiting the context, the attribute `output` will have the logging output.

    Sample usage:

            >>> with CaptureLog() as cl:
            ...     cl.logger.info('Do some stuff')
            >>> assert cl.output == 'Do some stuff'

    """
    def __init__(self, level=3):
        from io import StringIO
        import logging

        logging_levels = { 0: logging.CRITICAL,
                           1: logging.WARNING,
                           2: logging.INFO,
                           3: logging.DEBUG }
        self.logger = logging.getLogger('CaptureLog')
        self.logger.setLevel(logging_levels[level])
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.handler.flush()
        self.output = self.stream.getvalue().strip()
        self.handler.close()


@contextmanager
def assert_no_error_logs(error_level=logging.ERROR,
                         logger_level=logging.ERROR):
    """Context manager, which provides an `InMemoryLogger` instance and checks,
    that no records with a level higher or equal to `level` have been
    logged in the body of the manager.
    """
    logger = InMemoryLogger(logger_level)
    yield logger
    error_records = [
        rec for rec in logger.records if rec.levelno >= error_level
    ]
    if error_records:
        error_messages = "\n".join(
            logging.getLevelName(rec.levelno) + ": " +
            logger.formatter.format(rec) for rec in error_records)
        msg = f"Logged messages above or equal to error_level = {error_level}:\n{error_messages}"
        raise AssertionError(msg)


class InMemoryLogger(logging.Logger):
    """Logger which buffers messages in memory.
    Logged messages are available ungrouped in the attribute `messages`
    and grouped by level in `messages_by_level`.
    If more metadata is needed, raw records are available in `records`.
    """

    def __init__(self, level=logging.INFO):
        super().__init__(type(self).__name__, level)
        self.messages_by_level = {}
        self.messages = []
        self.records = []
        self.formatter = logging.Formatter()

    def handle(self, record):
        msg = self.formatter.format(record)
        message_buffer = self.messages_by_level.setdefault(record.levelno, [])
        message_buffer.append(msg)
        self.messages.append(msg)
        self.records.append(record)
        print(f"[{record.levelno}]", msg)
