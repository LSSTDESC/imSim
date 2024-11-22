import numpy as np

from random import shuffle
from dataclasses import replace
from collections import Counter

from galsim.config import valid_image_types
from imsim.stamp import ObjectCache, ProcessingMode

def create_fft_obj_list(num_objects, flux=1e6, start_num=0):
    return [ObjectCache(i+start_num, flux, ProcessingMode.FFT) for i in range(num_objects)]

def create_phot_obj_list(num_objects, flux=1e5, start_num=0):
    return [ObjectCache(i+start_num, flux, ProcessingMode.PHOT) for i in range(num_objects)]

def create_faint_obj_list(num_objects, flux=100, start_num=0):
    return [ObjectCache(i+start_num, flux, ProcessingMode.FAINT) for i in range(num_objects)]

def create_mixed_obj_list():
    # FFT, photon and faint photon objects.
    base_list = create_fft_obj_list(10) + create_phot_obj_list(9) + create_faint_obj_list(1)
    # Shuffle the objects then re-index them.
    shuffle(base_list)
    base_list = [replace(object, index=i) for i, object in enumerate(base_list)]
    return base_list

def run_partition_all_same_object_type(create_list_fn, desired_mode):
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    n_obj = 20
    nbatch = 10  # Not important but need to be passed to partition.
    orig_objects = create_list_fn(n_obj)
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        orig_objects,
        nbatch,
    )
    # Assert correct number of objects in each list.
    if desired_mode == ProcessingMode.FFT:
        assert len(fft_objects) == n_obj
        assert len(phot_objects) == 0
        assert len(faint_objects) == 0
        objects = fft_objects
    elif desired_mode == ProcessingMode.PHOT:
        assert len(fft_objects) == 0
        assert len(phot_objects) == n_obj
        assert len(faint_objects) == 0
        objects = phot_objects
    elif desired_mode == ProcessingMode.FAINT:
        assert len(fft_objects) == 0
        assert len(phot_objects) == 0
        assert len(faint_objects) == n_obj
        objects = faint_objects
    # Assert that all objects in the original list appear once in the new list.
    counts = Counter(object.index for object in objects)
    assert all([counts[obj.index] == 1 for obj in orig_objects])
    # and that they all have the expected draw mode.
    assert all([object.mode == desired_mode for object in objects])
    return

def test_partition_objects_all_fft():
    run_partition_all_same_object_type(create_fft_obj_list, ProcessingMode.FFT)

def test_partition_objects_all_phot():
    run_partition_all_same_object_type(create_phot_obj_list, ProcessingMode.PHOT)

def test_partition_objects_all_faint():
    run_partition_all_same_object_type(create_faint_obj_list, ProcessingMode.FAINT)

def test_partition_objects_mixed_all_types():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    nbatch = 10
    orig_objects = create_mixed_obj_list()
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        orig_objects,
        nbatch,
    )
    # Assert correct number of objects in each list.
    assert len(fft_objects) == 10
    assert len(phot_objects) == 9
    assert len(faint_objects) == 1
    # Assert that all objects in the original list appear once in the new list.
    all_objects = fft_objects + phot_objects + faint_objects
    counts = Counter(object.index for object in all_objects)
    assert all([counts[obj.index] == 1 for obj in orig_objects])
    expected_mode = [ProcessingMode.FFT] * 10 + [ProcessingMode.PHOT] * 9 + [ProcessingMode.FAINT]
    assert all([object.mode == expected_mode[i] for i, object in enumerate(all_objects)])
    
    return

def test_partition_objects_photon_and_faint():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    nbatch = 100
    n_obj_phot = 10
    n_obj_faint = 5
    # We also want some PHOT objects (so they get proper SED treatment) but
    # which have low enough flux that they should be batched as if FAINT.
    n_obj_phot_low_flux = 5
    flux_phot_low_flux = 50

    orig_objects = (create_phot_obj_list(n_obj_phot, start_num=0) +
                    create_faint_obj_list(n_obj_faint, start_num=n_obj_phot) +
                    create_phot_obj_list(n_obj_phot_low_flux, flux=flux_phot_low_flux,start_num=n_obj_phot+n_obj_faint))

    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        orig_objects,
        nbatch,
    )

    # Assert expected number of objects in each list.
    assert len(fft_objects) == 0
    assert len(phot_objects) == 10
    assert len(faint_objects) == 10

    # All objects in phot_objects should have ProcessingMod.PHOT and
    # phot_flux >= nbatch.
    assert all([object.mode == ProcessingMode.PHOT and object.phot_flux >= nbatch for object in phot_objects])

    # All objects in faint_objects should have ProcessingMode.FAINT
    # or (ProcessingMode.PHOT and phot_flux < nbatch).
    assert all([object.mode == ProcessingMode.FAINT or (object.mode == ProcessingMode.PHOT and object.phot_flux < nbatch) for object in faint_objects])

    # Assert that all the original objects appear in the recombined partitioned
    # lists once only.
    all_objects = fft_objects + phot_objects + faint_objects
    counts = Counter(object.index for object in all_objects)
    assert all([counts[obj.index] == 1 for obj in orig_objects])

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
        assert all([object.index == batch_index * nobjects_per_batch + i for i, object in enumerate(batch)])

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
        assert all(object.index == nobj_previous + i for i, object in enumerate(batch))
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
        assert all(object.index == nobj_previous + i for i, object in enumerate(batch))
        nobj_previous += len(batch)

    return

def test_make_photon_batches():
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    n_obj_phot = 15
    n_obj_faint = 5
    nobjects = n_obj_phot + n_obj_faint
    phot_objects = create_phot_obj_list(n_obj_phot, start_num=0)
    faint_objects = create_faint_obj_list(n_obj_faint, start_num=n_obj_phot)
    objects = phot_objects + faint_objects
    orig_flux = np.empty(nobjects)
    for i, object in enumerate(objects):
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
