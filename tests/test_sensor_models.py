import os
import numpy as np

import galsim
from imsim.meta_data import data_dir

SENSOR_DIR = os.path.join(data_dir, 'sensor_models')
NX = 17
NY = 17
SCALE = 0.3

def sensor_path(name):
    return os.path.join(SENSOR_DIR, name)

def draw_obj_with_sensor(obj, sensor, rng):
    image = obj.drawImage(sensor=sensor, method='phot', poisson_flux=False, rng=rng, nx=NX, ny=NY, scale=SCALE)
    image.setCenter(0, 0)
    return image

def run_sensor_tests(sensor_type):

    if sensor_type not in ['itl', 'e2v']:
        raise ValueError(f"Unrecognized sensor type: {sensor_type}")

    rng0 = galsim.BaseDeviate(1234)
    rng1 = galsim.BaseDeviate(1234)
    rng2 = galsim.BaseDeviate(1234)
    rng3 = galsim.BaseDeviate(1234)
    rng4 = galsim.BaseDeviate(1234)

    # Create a very small and bright object.
    obj = galsim.Gaussian(flux=1.e6, sigma=0.3)

    # Without a sensor.
    image0 = draw_obj_with_sensor(obj, None, rng0)

    # With a simple sensor which has no electrostatic effects.
    sensor1 = galsim.Sensor()
    image1 = draw_obj_with_sensor(obj, sensor1, rng1)

    # With the four point sensor model.
    sensor2 = galsim.SiliconSensor(rng=rng2, name=sensor_path('lsst_' + sensor_type + '_50_4'))
    image2 = draw_obj_with_sensor(obj, sensor2, rng2)

    # With the eight point sensor model.
    sensor3 = galsim.SiliconSensor(rng=rng3, name=sensor_path('lsst_' + sensor_type + '_50_8'))
    image3 = draw_obj_with_sensor(obj, sensor3, rng3)

    # With the 32 point sensor model.
    sensor4 = galsim.SiliconSensor(rng=rng4, name=sensor_path('lsst_' + sensor_type + '_50_32'))
    image4 = draw_obj_with_sensor(obj, sensor4, rng4)

    r0 = image0.calculateMomentRadius(flux=obj.flux)
    r1 = image1.calculateMomentRadius(flux=obj.flux)
    r2 = image2.calculateMomentRadius(flux=obj.flux)
    r3 = image3.calculateMomentRadius(flux=obj.flux)
    r4 = image4.calculateMomentRadius(flux=obj.flux)

    print('Flux = %.0f:  sum        peak          radius' % obj.flux)
    print('im0:         %.1f     %.2f       %f' % (image0.array.sum(),image0.array.max(), r0))
    print('im1:         %.1f     %.2f       %f' % (image1.array.sum(),image1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f' % (image2.array.sum(),image2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f' % (image3.array.sum(),image3.array.max(), r3))
    print('im4:         %.1f     %.2f       %f' % (image4.array.sum(),image4.array.max(), r4))

    # The max flux in the images using a SiliconSensor model should always be
    # less than or equal to the flux when not using a sensor or using the simple Sensor.
    assert image2.array.max() <= image0.array.max()
    assert image3.array.max() <= image0.array.max()
    assert image4.array.max() <= image0.array.max()
    assert image2.array.max() <= image1.array.max()
    assert image3.array.max() <= image1.array.max()
    assert image4.array.max() <= image1.array.max()

    # The spot's radius in the images using the simple Sensor should always be
    # less than or equal to the radius from using a SiliconSensor model.
    assert r0 <= r2
    assert r0 <= r3
    assert r0 <= r4
    assert r1 <= r2
    assert r1 <= r3
    assert r1 <= r4

    # Following the GalSim silicon sensor tests and docs:
    sigma_r = 1. / np.sqrt(obj.flux) * image0.scale
    # Firstly, images with None and the simple Sensor should have consistently
    # sized spots.
    print('check |r1-r0| = %f <? %f' % (np.abs(r1-r0), 2.*sigma_r))
    np.testing.assert_allclose(r0, r1, atol=2.*sigma_r)
    # Then, the spots in the images using the SiliconSensor models should all be
    # larger than the None/Sensor images.
    print('check r2 - r0 = %f > %f due to brighter-fatter' % (r2-r0,2*sigma_r))
    assert r2 - r0 > 2 * sigma_r
    print('check r3 - r0 = %f > %f due to brighter-fatter' % (r3-r0,2*sigma_r))
    assert r3 - r0 > 2 * sigma_r
    print('check r4 - r0 = %f > %f due to brighter-fatter' % (r4-r0,2*sigma_r))
    assert r4 - r0 > 2 * sigma_r
    # Finally, the different SiliconSensor models, which each use a different
    # number of vertices in the pixel models, should have spots of consistent
    # sizes.
    print('check |r3-r2| = %f <? %f' % (np.abs(r3-r2), 2.*sigma_r))
    np.testing.assert_allclose(r3, r2, atol=2.*sigma_r)
    print('check |r4-r3| = %f <? %f' % (np.abs(r4-r3), 2.*sigma_r))
    np.testing.assert_allclose(r4, r3, atol=2.*sigma_r)

def test_itl_sensor():
    """Test the ITL sensor models."""
    print("--- Testing ITL sensor models ---")
    run_sensor_tests('itl')

def test_e2v_sensor():
    """Test the e2v sensor models."""
    print("--- Testing e2v sensor models ---")
    run_sensor_tests('e2v')


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
