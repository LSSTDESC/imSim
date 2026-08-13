import numpy as np

from random import shuffle
from dataclasses import replace
from collections import Counter

from galsim.config import valid_image_types
from imsim.stamp import ObjectInfo, ProcessingMode

def create_fft_obj_list(num_objects, flux=1e6, start_num=0):
    return [ObjectInfo(i+start_num, flux, ProcessingMode.FFT) for i in range(num_objects)]

def create_phot_obj_list(num_objects, flux=1e5, start_num=0):
    return [ObjectInfo(i+start_num, flux, ProcessingMode.PHOT) for i in range(num_objects)]

def create_faint_obj_list(num_objects, flux=100, start_num=0):
    return [ObjectInfo(i+start_num, flux, ProcessingMode.FAINT) for i in range(num_objects)]

def shuffle_batch(batch):
    shuffle(batch)
    batch = [replace(object, index=i) for i, object in enumerate(batch)]
    return batch

def create_mixed_obj_list():
    # FFT, photon and faint photon objects.
    base_list = create_fft_obj_list(10) + create_phot_obj_list(9) + create_faint_obj_list(1)
    base_list = shuffle_batch(base_list)
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
    print("Original objects:")
    for obj in orig_objects:
        print(f" Index: {obj.index}, flux: {obj.phot_flux}, mode: {obj.mode}")
    fft_objects, phot_objects, faint_objects = builder.partition_objects(
        orig_objects,
        nbatch,
    )
    print("FFT objects:")
    for obj in fft_objects:
        print(f" Index: {obj.index}, flux: {obj.phot_flux}, mode: {obj.mode}")
    print("photon objects:")
    for obj in phot_objects:
        print(f" Index: {obj.index}, flux: {obj.phot_flux}, mode: {obj.mode}")
    print("faint objects:")
    for obj in faint_objects:
        print(f" Index: {obj.index}, flux: {obj.phot_flux}, mode: {obj.mode}")
    # Assert correct number of objects in each list.
    assert len(fft_objects) == 10
    assert len(phot_objects) == 9
    assert len(faint_objects) == 1
    # Assert that all objects in the original list appear once in the new list.
    all_objects = fft_objects + phot_objects + faint_objects
    counts = Counter(object.index for object in all_objects)
    print("Counts:", counts)
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


def run_subbatch_test(name, batch, nsubbatch):
    total_original_flux = sum(obj.phot_flux for obj in batch)
    print('total original flux = ',total_original_flux)
    builder = valid_image_types["LSST_PhotonPoolingImage"]
    subbatches = builder.make_photon_subbatches(batch, nsubbatch)
    print('total subbatch flux = ',sum([sum(obj.phot_flux for obj in subbatch) for subbatch in subbatches]))

    # Show what the subbatches contain:
    #   index  total_flux   count
    # For small subbatches, list the individual objects.
    print("Subbatches in test:", name)
    for i, subbatch in enumerate(subbatches):
        total_flux = sum(obj.phot_flux for obj in subbatch)
        print(f" Subbatch {i}: {total_flux} {len(subbatch)}")
        if len(subbatch) <= 5:
            print(f"     {[ (obj.index, obj.phot_flux) for obj in subbatch ]}")
    nsplit = sum(len(subbatch) for subbatch in subbatches) - len(batch)
    print(f"{nsplit} objects were split across multiple subbatches (with multiplicity)")

    # Check that the correct number of batches was created.
    assert len(subbatches) == nsubbatch

    # Check that each object's flux preserved across all subbatches in which it appears.
    for orig_obj in batch:
        total_obj_flux = sum(sum(obj.phot_flux for obj in subbatch if obj.index == orig_obj.index)
                             for subbatch in subbatches)
        assert np.isclose(orig_obj.phot_flux, total_obj_flux)

    # Check that the total flux of subbatches is the same as the total flux in the original.
    assert np.isclose(sum([sum(obj.phot_flux for obj in subbatch) for subbatch in subbatches]),
                      total_original_flux)

    # Check that all components have positive flux.
    assert all(obj.phot_flux > 0 for subbatch in subbatches for obj in subbatch)

    # Check that the subbatches all have about the same number of photons each.
    # This assert isn't formally required by the algorithm, but in practice, it seems
    # to always do a good job now.
    total_subbatch_fluxes = [sum(obj.phot_flux for obj in subbatch) for subbatch in subbatches]
    assert max(total_subbatch_fluxes) <= 1.02 * min(total_subbatch_fluxes)


def test_make_photon_subbatches():
    """
    Test the newer sub-batching method which attempts to spread the batch
    flux equally across the sub-batches.
    Some of these tests may be too restrictive, in particular those which specify
    the exact sub-batch contents. It may be better to set those aside on only
    check that the sub-batches balance flux and contain all the objects with the
    correct total flux.
    """
    # Create a batch with a 'large' number of objects with varying fluxes, but
    # with none dominating the total. We want them to be spread across the
    # sub-batches s.t. each has a flux of 2e4.
    batch = [ObjectInfo(0, 1e4, ProcessingMode.PHOT),
             ObjectInfo(1, 5e3, ProcessingMode.PHOT),
             ObjectInfo(2, 2e3, ProcessingMode.PHOT),
             ObjectInfo(3, 7e3, ProcessingMode.PHOT),
             ObjectInfo(4, 3e3, ProcessingMode.PHOT),
             ObjectInfo(5, 5e3, ProcessingMode.PHOT),
             ObjectInfo(6, 1e4, ProcessingMode.PHOT),
             ObjectInfo(7, 2e4, ProcessingMode.PHOT),
             ObjectInfo(8, 1e4, ProcessingMode.PHOT),
             ObjectInfo(9, 8e3, ProcessingMode.PHOT),
             ]
    run_subbatch_test("equal distribution", batch, 4)

    # Create a batch with a total flux of 1e6 photons. We should end up with 10
    # sub-batches of 1e5 photons each. The majority of the flux is in a few very
    # bright objects, so we want to see these correctly being split up across
    # multiple sub-batches to make sure no one sub-batch requires a lot of
    # time/memory. Yes, we're doing more work in the background for the extra
    # objects, but this should be very little work as long as it's only for a
    # very few bright objects.
    batch = [ObjectInfo(0, 6e5, ProcessingMode.PHOT),
             ObjectInfo(1, 2e5, ProcessingMode.PHOT),
             ObjectInfo(2, 1e5, ProcessingMode.PHOT),
             ObjectInfo(3, 4e4, ProcessingMode.PHOT),
             ObjectInfo(4, 2e4, ProcessingMode.PHOT),
             ObjectInfo(5, 1e4, ProcessingMode.PHOT),
             ObjectInfo(6, 1e4, ProcessingMode.PHOT),
             ObjectInfo(7, 1e4, ProcessingMode.PHOT),
             ObjectInfo(8, 5e3, ProcessingMode.PHOT),
             ObjectInfo(9, 5e3, ProcessingMode.PHOT),
             ]
    run_subbatch_test("bright object fragmentation", batch, 10)

    # Here there's still one very bright object, but it leaves a little bit of
    # space in the first subbatch for something else to go in. The other faint
    # objects pack into the second one.
    batch = [ObjectInfo(0, 8e5, ProcessingMode.PHOT),
             ObjectInfo(1, 3e5, ProcessingMode.PHOT),
             ObjectInfo(2, 6e5, ProcessingMode.PHOT),
             ObjectInfo(3, 3e5, ProcessingMode.PHOT),
             ]
    run_subbatch_test("fragmentation in first sub-batch", batch, 2)

    # Make sure the sub-batcher can go backwards (i.e. assign to subbatches
    # earlier then the one just filled). This would be important for best fit
    # type implementations, but in other implementations like first fit might be
    # equivalent to the previous test.
    batch = [ObjectInfo(0, 8e4, ProcessingMode.PHOT),
             ObjectInfo(1, 8e4, ProcessingMode.PHOT),
             ObjectInfo(2, 4e4, ProcessingMode.PHOT),
             ]
    run_subbatch_test("filling early sub-batches", batch, 2)


def test_make_photon_subbatches_non_simple():
    # Need a test for which division of flux across sub-batches is not even,
    # requiring non-trivial fragmentation of objects.

    # The first test places a total of 1e6 photons across 7 sub-batches,
    # i.e. 1e6 mod 7 = 142857 photons per sub-batch with remainder 1.
    batch = [ObjectInfo(0, 5e5, ProcessingMode.PHOT),
             ObjectInfo(1, 5e5, ProcessingMode.PHOT),
             ]
    run_subbatch_test("small non-simple fragmentation", batch, 7)

    # Then place 3 objects with total flux 1.1e6 in 31 sub-batches,
    # i.e. 35483 photons per sub-batch with remainder 27.
    batch = [ObjectInfo(0, 1e5, ProcessingMode.PHOT),
             ObjectInfo(1, 5e5, ProcessingMode.PHOT),
             ObjectInfo(2, 5e5, ProcessingMode.PHOT),
             ]
    run_subbatch_test("large non-simple fragmentation", batch, 31)

def test_make_photon_subbatches_power_law():
    # This is intended to be a semi-realistic flux distribution with a fairly large number
    # of objects. We use a power law:
    #     p(flux) ~ flux^-s
    # which is relatively common for flux distributions of real objects with a power
    # law index near 1.

    nobj = 2000
    nbatch = 200
    s = 0.95   # power law index must be < 1.
    fmin = 1.
    fmax = 1.e6

    # This distribution is realized by inverting the cdf of the this distribution
    # p = A (f/fmax)^-s
    # cdf = A/(1-s) [f^(1-s) - fmin^(1-s)] / fmax^-s
    #
    # Solve for A from cdf(f=fmax) = 1
    # A/(1-s) = fmax^-1
    #
    # -> cdf = (f/fmax)^(1-s)
    # f/fmax = cdf^(1/(1-s))
    #
    # To keep the objects having f > fmin, we clip the cdf to the range that gives f>fmin.
    cdf_min = (fmin/fmax)**(1-s)
    rng = np.random.default_rng(1234)
    cdf = rng.uniform(cdf_min, 1, size=nobj)
    f = fmax * cdf**(1/(1-s))
    np.testing.assert_array_less(0.99, f)
    print('Total flux = ',np.sum(f))
    print('Brightest obj flux = ',np.max(f))
    assert(np.max(f) > np.sum(f) / nbatch)  # I.e. we want at least one obj to need to split.
    batch = [ObjectInfo(i, f[i], ProcessingMode.PHOT) for i in range(nobj)]
    run_subbatch_test("large power law distribution", batch, nbatch)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
