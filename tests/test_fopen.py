"Unit tests for fopen function"

import os
import unittest
import gzip
import imsim

class FopenTestCase(unittest.TestCase):
    "TestCase class for fopen unit tests."
    def setUp(self):
        self.test_dir = 'fopen_dir'
        os.makedirs(self.test_dir, exist_ok=True)
        self.fopen_test_file \
            = os.path.join(self.test_dir, 'fopen_test_file.txt')
        self.fopen_include_file1 \
            = os.path.join(self.test_dir, 'fopen_include_file1.txt.gz')
        self.fopen_include_file2 \
            = os.path.join(self.test_dir, 'fopen_include_file2.txt.gz')
        self.lines = 'line1 line2 line3 line4'.split()
        with open(self.fopen_test_file, 'w') as output:
            output.write('%s\n' % self.lines[0])
            output.write('%s\n' % self.lines[1])
            output.write('includeobj %s\n'
                         % os.path.basename(self.fopen_include_file1))
            output.write('includeobj %s\n'
                         % os.path.basename(self.fopen_include_file2))
        with gzip.open(self.fopen_include_file1, 'wt') as output:
            output.write('%s\n' % self.lines[2])
        with gzip.open(self.fopen_include_file2, 'wt') as output:
            output.write('%s\n' % self.lines[3])

    def tearDown(self):
        for item in (self.fopen_test_file, self.fopen_include_file1,
                     self.fopen_include_file2):
            try:
                os.remove(item)
            except OSError:
                pass
        try:
            os.rmdir(self.test_dir)
        except OSError:
            pass

    def test_fopen(self):
        "Test the fopen function."
        with imsim.fopen(self.fopen_test_file, mode='rt') as input_:
            for i, line in enumerate(input_):
                expected = self.lines[i]
                self.assertEqual(line.strip(), expected)
            # Make sure all of the expected lines have been processed.
            self.assertEqual(self.lines[-1], expected)

if __name__ == '__main__':
    unittest.main()
