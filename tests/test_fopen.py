"Unit tests for fopen function"

import os
import unittest
import gzip
import desc.imsim

class FopenTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'fopen_dir'
        os.makedirs(self.test_dir, exist_ok=True)
        self.fopen_test_file \
            = os.path.join(self.test_dir, 'fopen_test_file.txt')
        self.fopen_include_file \
            = os.path.join(self.test_dir, 'fopen_include_file.txt.gz')
        self.lines = 'line1 line2 line3'.split()
        with open(self.fopen_test_file, 'w') as output:
            output.write('%s\n' % self.lines[0])
            output.write('%s\n' % self.lines[1])
            output.write('includeobj %s\n'
                         % os.path.basename(self.fopen_include_file))
        with gzip.open(self.fopen_include_file, 'wt') as output:
            output.write('%s\n' % self.lines[2])

    def tearDown(self):
        for item in (self.fopen_test_file, self.fopen_include_file):
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
        with desc.imsim.fopen(self.fopen_test_file, mode='rt') as input_:
            for i, line in enumerate(input_):
                self.assertEqual(line.strip(), self.lines[i])

if __name__ == '__main__':
    unittest.main()
