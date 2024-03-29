"""Unit tests for the InstCatalog class."""

from unittest import mock, TestCase

from imsim import instcat


@mock.patch("imsim.instcat.fopen", mock.mock_open(read_data=""))
@mock.patch("imsim.instcat.get_radec_limits", mock.Mock(return_value=(0.0,) * 9))
def test_empty_inst_catalog_raises_on_get_obj(*_mocks):
    catalog = instcat.InstCatalog("/tmp/testcat.txt", wcs=mock.Mock())
    with TestCase().assertRaises(RuntimeError):
        catalog.getObj(0)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
