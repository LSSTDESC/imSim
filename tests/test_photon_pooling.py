import numpy as np

from random import shuffle

from galsim.config import valid_image_types
from imsim.stamp import ObjectCache, ProcessingMode

def create_fft_obj_list(num_objects, flux=1e6):
    return [ObjectCache(i, flux, ProcessingMode.FFT) for i in range(num_objects)]

def create_phot_obj_list(num_objects, flux=1e5):
    return [ObjectCache(i, flux, ProcessingMode.PHOT) for i in range(num_objects)]

def create_faint_obj_list(num_objects, flux=100):
    return [ObjectCache(i, flux, ProcessingMode.FAINT) for i in range(num_objects)]

def create_mixed_obj_list():
    # FFT, photon and faint photon objects.
    base_list = create_fft_obj_list(10) + create_phot_obj_list(9) + create_faint_obj_list(1)
    shuffle(base_list)
    for i, object in enumerate(base_list):
        object.index = i
    return base_list

def test_partition_objects_all_fft():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        create_fft_obj_list(20)
    )
    assert len(fft_objects) == 20
    assert len(phot_objects) == 0
    assert len(faint_objects) == 0
    return

def test_partition_objects_all_photon():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        create_phot_obj_list(20)
    )
    assert len(fft_objects) == 0
    assert len(phot_objects) == 20
    assert len(faint_objects) == 0
    return

def test_partition_objects_all_faint():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        create_faint_obj_list(20)
    )
    assert len(fft_objects) == 0
    assert len(phot_objects) == 0
    assert len(faint_objects) == 20
    return

def test_partition_objects_mixed_types():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        create_mixed_obj_list()
    )
    assert len(fft_objects) == 10
    assert len(phot_objects) == 9
    assert len(faint_objects) == 1
    return

def test_make_batches():
    """Test make_batches which is only used with FFT
    objects."""
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    objects = create_fft_obj_list(20)

    # The first case is simple: place 20 objects in 10 batches.
    # Each batch should contain 2 objects.
    nobjects = 20
    nbatch = 10
    nobjects_per_batch = nobjects // nbatch
    # Generate and loop through the batches.
    for batch_index, batch in enumerate(builder.make_batches(objects, nbatch)):
        # Assert that the batch contains the expected number of objects.
        assert len(batch) == nobjects_per_batch
        # Assert that all objects are present and in the expected order.
        for i in range(len(batch)):
            assert batch[i].index == batch_index * nobjects_per_batch + i

    # The second case places the same 20 objects in 6 batches. We expect 2
    # batches of 4 followed by 4 batches of 3.
    nbatch = 6
    nobj_previous = 0
    for batch_index, batch in enumerate(builder.make_batches(objects, nbatch)):
        # Assert that the correct number of objects are in this batch.
        if batch_index < 2:
            expected_nobjects_per_batch = 4
        else:
            expected_nobjects_per_batch = 3
        assert len(batch) == expected_nobjects_per_batch
        # Assert that all objects are present and in the expected order.
        for i in range(len(batch)):
            assert batch[i].index == nobj_previous + i
        nobj_previous += len(batch)

    # The final case places 20 objects in 7 batches. We now expect 2 batches of
    # 3 followed by 7 batches of 2.
    nbatch = 9
    nobj_previous = 0
    for batch_index, batch in enumerate(builder.make_batches(objects, nbatch)):
        # Assert that the correct number of objects are in this batch.
        if batch_index < 2:
            expected_nobjects_per_batch = 3
        else:
            expected_nobjects_per_batch = 2
        assert len(batch) == expected_nobjects_per_batch
        # Assert that all objects are present and in the expected order.
        for i in range(len(batch)):
            assert batch[i].index == nobj_previous + i
        nobj_previous += len(batch)

    return

def test_make_photon_batches():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    n_obj_phot = 15
    n_obj_faint = 5
    nobjects = n_obj_phot + n_obj_faint
    phot_objects = create_phot_obj_list(n_obj_phot)
    faint_objects = create_faint_obj_list(n_obj_faint)
    objects = phot_objects + faint_objects
    orig_flux = np.empty(nobjects)
    for i, object in enumerate(objects):
        object.index = i
        orig_flux[i] = object.phot_flux

    # Create 11 batches to ensure things don't divide nicely.
    nbatch = 11
    batches = builder.make_photon_batches({}, {}, None, phot_objects, faint_objects, nbatch)

    # Count how many times the objects appear in the batches and sum their total
    # flux across all batches.
    count = [0] * nobjects
    total_flux = np.zeros(nobjects)#, dtype="i8")
    for batch in batches:
        for object in batch:
            count[object.index] += 1
            total_flux[object.index] += object.phot_flux

    # Assert that the PHOT objects appear in all batches (This may not be
    # correct in the future if PHOT objects are spread across subsets of batches
    # rather than all of them.)
    # Also assert that FAINT objects appear once and only once.
    for i, object in enumerate(objects):
        if object.mode == ProcessingMode.PHOT:
            assert count[i] == nbatch
        elif object.mode == ProcessingMode.FAINT:
            assert count[i] == 1

    # Assert the summed flux across the objects in the batches is correct.
    np.testing.assert_array_almost_equal(total_flux, orig_flux)

if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
