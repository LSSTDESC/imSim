from astropy.utils import iers
import imsim

def test_iers_download_disabled():
    assert not iers.conf.auto_download

if __name__ == '__main__':
    test_iers_download_disabled()
