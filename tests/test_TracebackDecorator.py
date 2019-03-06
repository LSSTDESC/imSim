"""
Unit tests for TracebackDecorator.
"""
import os
import unittest
import multiprocessing
import contextlib
import desc.imsim


def call_back_function(item):
    """Call back function that raises an exception for a specific item."""
    if item == 1:
        raise RuntimeError("item == 1")


def context_function(processes=5, apply_decorator=True):
    """
    Helper function to run multiprocessing jobs in a
    contextlib.redirect_stderr context with and without using a
    TracebackDecorator.
    """
    with multiprocessing.Pool(processes=processes) as pool:
        if apply_decorator:
            func = desc.imsim.TracebackDecorator(call_back_function)
        else:
            func = call_back_function

        workers = [pool.apply_async(func, (item,)) for item in range(processes)]
        pool.close()
        pool.join()
        return [worker.get() for worker in workers]


class TracebackDecoratorTestCase(unittest.TestCase):
    """
    TestCase class for desc.imsim.TracebackDecorator.
    """
    def setUp(self):
        self.filename = 'test_traceback_decorator.txt'

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def test_TracebackDecorator(self):
        """Test the TracebackDecorator class."""

        # Test that the expected exception message does appear in the
        # output when using the TracebackDecorator.
        with open(self.filename, 'w') as output:
            with contextlib.redirect_stdout(output),\
                 contextlib.redirect_stderr(output):
                try:
                    context_function(apply_decorator=True)
                except Exception:
                    pass
        with open(self.filename, 'r') as input_:
            lines = [_.strip() for _ in input_]
        self.assertTrue('RuntimeError: item == 1' in lines)

        # Test that the expected exception message is not in the
        # output without using the TracebackDecorator.
        with open(self.filename, 'w') as output:
            with contextlib.redirect_stdout(output),\
                 contextlib.redirect_stderr(output):
                try:
                    context_function(apply_decorator=False)
                except Exception:
                    pass
        with open(self.filename, 'r') as input_:
            lines = [_.strip() for _ in input_]
        self.assertFalse('RuntimeError: item == 1' in lines)


if __name__ == '__main__':
    unittest.main()
