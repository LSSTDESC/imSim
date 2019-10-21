"""
Unit test for get_stack_products function.
"""
import unittest
import subprocess
from desc.imsim import get_stack_products


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
        target_lists = [None, 'afw sims_photUtils sims_utils'.split()]

        for targets in target_lists:
            products = get_stack_products(targets)
            if targets is not None:
                self.assertEqual(set(products.keys()), set(targets))
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


if __name__ == '__main__':
    unittest.main()
