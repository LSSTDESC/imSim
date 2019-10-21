"""
Unit test for get_stack_products function.
"""
import unittest
import subprocess
from collections import defaultdict
import desc.imsim


class GetStackProductsTestCase(unittest.TestCase):
    """
    TestCase subclass for testing get_stacks_products.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_stack_products(self):
        """Test the get_stack_products function."""
        target_dicts = [None, {_: None for _ in
                               'afw sims_photUtils sims_utils'.split()}]

        for targets in target_dicts:
            products = desc.imsim.get_stack_products(targets)
            if targets is not None:
                self.assertEqual(products.keys(), targets.keys())
            for target in products:
                # Test result against eups command line result.
                command = f'eups list {target} -s'
                line = subprocess.check_output(command, shell=True)
                tokens = line.decode('utf-8').strip().split()
                version = tokens[0]
                tags = set(tokens[1:])
                tags.remove('setup')

                self.assertEqual(products[target].version, version)
                self.assertEqual(set(products[target].tags), tags)

    def test_get_version_keywords(self):
        """Test the get_version_keywords function."""
        class KeywordInfo:
            """Class to hold FITS keyword name and type values for
            versioning keywords."""
            pass
        version_keywords = desc.imsim.get_version_keywords()
        keyword_info = defaultdict(KeywordInfo)
        for key, value in version_keywords.items():
            if key.startswith('PKG'):
                iprod = int(key[len('PKG'):])
                keyword_info[iprod].name = value
            if key.startswith('TAG'):
                iprod = int(key[len('TAG'):])
                keyword_info[iprod].type = 'metapackage'
            if key.startswith('VER'):
                iprod = int(key[len('VER'):])
                keyword_info[iprod].type = ''
        # Repackage the keyword_info dict to use package names instead
        # of the iprod index so that we can compare directly to the
        # 'stack_packages' config.
        keyword_info = {_.name: _.type for _ in keyword_info.values()}

        config = desc.imsim.get_config()
        self.assertEqual(keyword_info, config['stack_packages'])


if __name__ == '__main__':
    unittest.main()
