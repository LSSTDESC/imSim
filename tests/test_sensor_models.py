import os
import numpy as np

import galsim
from imsim.meta_data import data_dir

SENSOR_DIR = os.path.join(data_dir, 'sensor_models')
NX = 17
NY = 17
SCALE = 0.3

# Raw moments and ellipticities for use in regression tests.
ITL_MOMENTS = {'None': {'Mxx': np.float64(1.0814199384960002), 'Myy': np.float64(1.0829925551110002),
                        'e1': np.float64(-0.0007265789768101491), 'e2': np.float64(-0.0007908239178325261)},
               'Sensor': {'Mxx': np.float64(1.0814199384960002), 'Myy': np.float64(1.0829925551110002),
                          'e1': np.float64(-0.0007265789768101491), 'e2': np.float64(-0.0007908239178325261)},
               '4point': {'Mxx': np.float64(1.2904056635999999), 'Myy': np.float64(1.2986653947160003),
                          'e1': np.float64(-0.0031902295958507877), 'e2': np.float64(-0.0017601284079719517)},
               '8point': {'Mxx': np.float64(1.2903588210709998), 'Myy': np.float64(1.298329443484),
                          'e1': np.float64(-0.0030790197962945716), 'e2': np.float64(-0.0017759453492135085)},
               '32point': {'Mxx': np.float64(1.290387793884), 'Myy': np.float64(1.298246399375),
                           'e1': np.float64(-0.0030358115146065564), 'e2': np.float64(-0.001785998312175343)}
               }
E2V_MOMENTS = {'None': {'Mxx': np.float64(1.0814199384960002), 'Myy': np.float64(1.0829925551110002),
                        'e1': np.float64(-0.0007265789768101491), 'e2': np.float64(-0.0007908239178325261)},
               'Sensor': {'Mxx': np.float64(1.0814199384960002), 'Myy': np.float64(1.0829925551110002),
                          'e1': np.float64(-0.0007265789768101491), 'e2': np.float64(-0.0007908239178325261)},
               '4point': {'Mxx': np.float64(1.305061712704), 'Myy': np.float64(1.321133490204),
                          'e1': np.float64(-0.006119795467680209), 'e2': np.float64(-0.0016835133150438853)},
               '8point': {'Mxx': np.float64(1.3052209484710002), 'Myy': np.float64(1.319877330876),
                          'e1': np.float64(-0.005583174740659797), 'e2': np.float64(-0.0017415076090550004)},
               '32point': {'Mxx': np.float64(1.3050858704), 'Myy': np.float64(1.319152136959),
                           'e1': np.float64(-0.005360133691972639), 'e2': np.float64(-0.0018021730905267673)}
               }
REG_MOMENTS = {'itl': ITL_MOMENTS, 'e2v': E2V_MOMENTS}

MOMENT_TOL = 1.e-6

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

    # Without a sensor (should be equivalent to Sensor).
    image0 = draw_obj_with_sensor(obj, None, rng0)

    # With a simple Sensor (should be equivalent to None).
    sensor1 = galsim.Sensor()
    image1 = draw_obj_with_sensor(obj, sensor1, rng1)

    # With the four point SiliconSensor model.
    sensor2 = galsim.SiliconSensor(rng=rng2, name=sensor_path('lsst_' + sensor_type + '_50_4'))
    image2 = draw_obj_with_sensor(obj, sensor2, rng2)

    # With the eight point SiliconSensor model.
    sensor3 = galsim.SiliconSensor(rng=rng3, name=sensor_path('lsst_' + sensor_type + '_50_8'))
    image3 = draw_obj_with_sensor(obj, sensor3, rng3)

    # With the 32 point SiliconSensor model.
    sensor4 = galsim.SiliconSensor(rng=rng4, name=sensor_path('lsst_' + sensor_type + '_50_32'))
    image4 = draw_obj_with_sensor(obj, sensor4, rng4)

    # The max flux in the images using a SiliconSensor model should always be
    # less than or equal to the flux when not using a sensor or using the simple Sensor.
    assert image2.array.max() <= image0.array.max()
    assert image3.array.max() <= image0.array.max()
    assert image4.array.max() <= image0.array.max()
    assert image2.array.max() <= image1.array.max()
    assert image3.array.max() <= image1.array.max()
    assert image4.array.max() <= image1.array.max()

    # Calculate the radii of the spots in each image.

    r = {'None': image0.calculateMomentRadius(flux=obj.flux),
         'Sensor': image1.calculateMomentRadius(flux=obj.flux),
         '4point': image2.calculateMomentRadius(flux=obj.flux),
         '8point': image3.calculateMomentRadius(flux=obj.flux),
         '32point': image4.calculateMomentRadius(flux=obj.flux),
         }

    print('Flux = %.0f:    sum        peak          radius' % obj.flux)
    print('None        :     %.1f     %.2f       %f' % (image0.array.sum(),image0.array.max(), r['None']))
    print('Sensor      :     %.1f     %.2f       %f' % (image1.array.sum(),image1.array.max(), r['Sensor']))
    print('Silicon 4pt :     %.1f     %.2f       %f' % (image2.array.sum(),image2.array.max(), r['4point']))
    print('Silicon 8pt :     %.1f     %.2f       %f' % (image3.array.sum(),image3.array.max(), r['8point']))
    print('Silicon 32pt:     %.1f     %.2f       %f' % (image4.array.sum(),image4.array.max(), r['32point']))

    print("Check spot sizes:")
    # Following the GalSim silicon sensor tests and docs:
    sigma_r = 1. / np.sqrt(obj.flux) * image0.scale
    # Firstly, images with None and the simple Sensor should have consistently
    # sized spots.
    print('check |rSensor-rNone| = %f < %f model consistency' % (np.abs(r['Sensor']-r['None']), 2.*sigma_r))
    np.testing.assert_allclose(r['None'], r['Sensor'], atol=2.*sigma_r)
    # Then, the spots in the images using the SiliconSensor models should all be
    # larger than the None/Sensor images.
    print('check r4point - rNone = %f > %f due to brighter-fatter' % (r['4point']-r['None'],2*sigma_r))
    assert r['4point'] - r['None'] > 2 * sigma_r
    print('check r8point - rNone = %f > %f due to brighter-fatter' % (r['8point']-r['None'],2*sigma_r))
    assert r['8point'] - r['None'] > 2 * sigma_r
    print('check r32point - rNone = %f > %f due to brighter-fatter' % (r['32point']-r['None'],2*sigma_r))
    assert r['32point'] - r['None'] > 2 * sigma_r
    # The different SiliconSensor models, which each use a different
    # number of vertices in the pixel models, should have spots of consistent
    # sizes.
    print('check |r8point-r4point| = %f < %f Silicon model consistency' % (np.abs(r['8point']-r['4point']), 2.*sigma_r))
    np.testing.assert_allclose(r['8point'], r['4point'], atol=2.*sigma_r)
    print('check |r32point-r8point| = %f < %f Silicon model consistency' % (np.abs(r['32point']-r['8point']), 2.*sigma_r))
    np.testing.assert_allclose(r['32point'], r['8point'], atol=2.*sigma_r)

    # Calculate the moments of the spots in each image.
    moments = {'None': galsim.utilities.unweighted_moments(image0),
               'Sensor': galsim.utilities.unweighted_moments(image1),
               '4point': galsim.utilities.unweighted_moments(image2),
               '8point': galsim.utilities.unweighted_moments(image3),
               '32point': galsim.utilities.unweighted_moments(image4),
               }

    print("Check moments:")
    print('None        : Mxx = %f    Myy = %f' % (moments['None']['Mxx'], moments['None']['Myy']))
    print('Sensor      : Mxx = %f    Myy = %f' % (moments['Sensor']['Mxx'], moments['Sensor']['Myy']))
    print('Silicon 4pt : Mxx = %f    Myy = %f' % (moments['4point']['Mxx'], moments['4point']['Myy']))
    print('Silicon 8pt : Mxx = %f    Myy = %f' % (moments['8point']['Mxx'], moments['8point']['Myy']))
    print('Silicon 32pt: Mxx = %f    Myy = %f' % (moments['32point']['Mxx'], moments['32point']['Myy']))

    # Check that calculated moments are within tolerance of regression values.
    np.testing.assert_allclose(moments['None']['Mxx'], REG_MOMENTS[sensor_type]['None']['Mxx'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['None']['Myy'], REG_MOMENTS[sensor_type]['None']['Myy'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['Sensor']['Mxx'], REG_MOMENTS[sensor_type]['Sensor']['Mxx'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['Sensor']['Myy'], REG_MOMENTS[sensor_type]['Sensor']['Myy'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['4point']['Mxx'], REG_MOMENTS[sensor_type]['4point']['Mxx'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['4point']['Myy'], REG_MOMENTS[sensor_type]['4point']['Myy'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['8point']['Mxx'], REG_MOMENTS[sensor_type]['8point']['Mxx'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['8point']['Myy'], REG_MOMENTS[sensor_type]['8point']['Myy'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['32point']['Mxx'], REG_MOMENTS[sensor_type]['32point']['Mxx'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(moments['32point']['Myy'], REG_MOMENTS[sensor_type]['32point']['Myy'], rtol=MOMENT_TOL)

    # Calculate the ellipticities of the spots in each image.
    ellipticities = {'None': galsim.utilities.unweighted_shape(image0),
                     'Sensor': galsim.utilities.unweighted_shape(image1),
                     '4point': galsim.utilities.unweighted_shape(image2),
                     '8point': galsim.utilities.unweighted_shape(image3),
                     '32point': galsim.utilities.unweighted_shape(image4),
                     }

    print("Check ellipticities:")
    print('None        : e1 = %.14e    e2 = %.14e' % (ellipticities['None']['e1'], ellipticities['None']['e2']))
    print('Sensor      : e1 = %.14e    e2 = %.14e' % (ellipticities['Sensor']['e1'], ellipticities['Sensor']['e2']))
    print('Silicon 4pt : e1 = %.14e    e2 = %.14e' % (ellipticities['4point']['e1'], ellipticities['4point']['e2']))
    print('Silicon 8pt : e1 = %.14e    e2 = %.14e' % (ellipticities['8point']['e1'], ellipticities['8point']['e2']))
    print('Silicon 32pt: e1 = %.14e    e2 = %.14e' % (ellipticities['32point']['e1'], ellipticities['32point']['e2']))

    # Check that calculated ellipticities are within tolerance of regression values.
    np.testing.assert_allclose(ellipticities['None']['e1'], REG_MOMENTS[sensor_type]['None']['e1'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['None']['e2'], REG_MOMENTS[sensor_type]['None']['e2'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['Sensor']['e1'], REG_MOMENTS[sensor_type]['Sensor']['e1'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['Sensor']['e2'], REG_MOMENTS[sensor_type]['Sensor']['e2'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['4point']['e1'], REG_MOMENTS[sensor_type]['4point']['e1'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['4point']['e2'], REG_MOMENTS[sensor_type]['4point']['e2'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['8point']['e1'], REG_MOMENTS[sensor_type]['8point']['e1'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['8point']['e2'], REG_MOMENTS[sensor_type]['8point']['e2'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['32point']['e1'], REG_MOMENTS[sensor_type]['32point']['e1'], rtol=MOMENT_TOL)
    np.testing.assert_allclose(ellipticities['32point']['e2'], REG_MOMENTS[sensor_type]['32point']['e2'], rtol=MOMENT_TOL)


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
