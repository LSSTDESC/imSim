from tempfile import TemporaryDirectory

import astropy.units as u
import galsim
import imsim
import numpy as np
from astropy.table import QTable, Table
from imsim.table_row import RowData


def create_table():
    table = QTable()
    # For testing single and multi indexing
    table["idx"] = [0, 1, 2, 3]
    table["idx0"] = [0, 0, 1, 1]
    table["idx1"] = [0, 1, 0, 1]
    table["unitless"] = np.array(table["idx"], dtype=float)
    table["angle1"] = table["idx"] * u.arcsec
    table["angle2"] = table["angle1"].to(u.deg)
    # structured array column
    tilt_dtype = np.dtype([("rx", "<f8"), ("ry", "<f8")])
    table["tilt"] = np.array([(0, 1), (1, 2), (2, 3), (3, 4)], dtype=tilt_dtype)
    table["tilt"].unit = u.arcsec
    # Add columns with other types
    table["int"] = np.array(table["idx"], dtype=int)
    table["bool"] = np.array([True, False, True, False], dtype=bool)
    table["str"] = ["a", "b", "c", "d"]
    # And some other units
    table["length"] = table["idx"] * u.m
    # Structured array with length units
    shift_dtype = np.dtype([("dx", "<f8"), ("dy", "<f8"), ("dz", "<f8")])
    table["shift"] = np.array(
        [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)], dtype=shift_dtype
    )
    table["shift"].unit = u.m

    table.pprint_all()
    return table


def check_row_data(config, idx):
    # Read unitful column as Angle
    a1, safe1 = RowData({"field": "angle1"}, config, galsim.Angle)
    a2, safe2 = RowData({"field": "angle2"}, config, galsim.Angle)
    assert safe1 == safe2 == True
    np.testing.assert_allclose(a1.rad, a2.rad, rtol=0, atol=1e-15)
    np.testing.assert_allclose(a1.rad, idx * u.arcsec.to(u.rad), rtol=0, atol=1e-15)

    # Reading the unitful columns directly as floats is permitted.  They'll
    # come back unequal this time due to unit differences.
    a1, safe1 = RowData({"field": "angle1"}, config, float)
    a2, safe2 = RowData({"field": "angle2"}, config, float)
    assert safe1 == safe2 == True
    assert isinstance(a1, float)
    assert isinstance(a2, float)
    np.testing.assert_allclose(
        (a1 * galsim.arcsec).rad,
        (a2 * galsim.degrees).rad,
        rtol=0,
        atol=1e-15,
    )

    # We can specify a to_unit though to get them to both come back in,
    # e.g., radians.
    a1, safe1 = RowData({"field": "angle1", "to_unit": "rad"}, config, float)
    a2, safe2 = RowData({"field": "angle2", "to_unit": "rad"}, config, float)
    assert safe1 == safe2 == True
    assert isinstance(a1, float)
    assert isinstance(a2, float)
    np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(a1, idx * u.arcsec.to(u.rad), rtol=0, atol=1e-15)

    # Reading the unitless column as an Angle is permitted with from_unit.
    a, safe = RowData(
        {"field": "unitless", "from_unit": "arcsec"}, config, galsim.Angle
    )
    assert safe == True
    np.testing.assert_allclose(a.rad, idx * u.arcsec.to(u.rad), rtol=0, atol=1e-15)

    # Read the unitless column as initially arcsec then convert to rad
    a, safe = RowData(
        {"field": "unitless", "from_unit": "arcsec", "to_unit": "rad"},
        config,
        float,
    )
    assert safe == True
    np.testing.assert_allclose(a, idx * u.arcsec.to(u.rad), rtol=0, atol=1e-15)

    # Using from_unit with a unitful column will raise if the units don't
    # match.
    with np.testing.assert_raises(ValueError):
        a, safe = RowData(
            {"field": "angle1", "from_unit": "deg"}, config, galsim.Angle
        )
    a, safe = RowData(
        {"field": "angle1", "from_unit": "arcsec"}, config, galsim.Angle
    )

    # It's an error to try to convert an Angle to a different unit.
    with np.testing.assert_raises(ValueError):
        a, safe = RowData(
            {"field": "angle1", "to_unit": "deg"}, config, galsim.Angle
        )

    # Read a structured array subfield
    rx, safe1 = RowData({"field": "tilt", "subfield": "rx"}, config, galsim.Angle)
    ry, safe2 = RowData({"field": "tilt", "subfield": "ry"}, config, galsim.Angle)
    assert safe1 == safe2 == True
    np.testing.assert_allclose(rx.rad, idx * u.arcsec.to(u.rad), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        ry.rad, (idx + 1) * u.arcsec.to(u.rad), rtol=0, atol=1e-15
    )

    # Read the full structured array as a list
    # The config layer is not set up to handle lists of Angles, though, so
    # we have to interpret as floats directly.
    tilt, safe = RowData({"field": "tilt"}, config, list)
    assert safe == True
    assert len(tilt) == 2
    np.testing.assert_allclose(tilt[0], idx, rtol=0, atol=1e-15)
    np.testing.assert_allclose(tilt[1], (idx + 1), rtol=0, atol=1e-15)

    # Read the full structured array and convert to rad
    tilt, safe = RowData({"field": "tilt", "to_unit": "rad"}, config, list)
    assert safe == True
    assert len(tilt) == 2
    np.testing.assert_allclose(tilt[0], idx * u.arcsec.to(u.rad), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        tilt[1], (idx + 1) * u.arcsec.to(u.rad), rtol=0, atol=1e-15
    )

    # Read the int column
    i, safe = RowData({"field": "int"}, config, int)
    assert safe == True
    assert i == idx

    # Read the bool column
    b, safe = RowData({"field": "bool"}, config, bool)
    assert safe == True
    assert b == (idx % 2 == 0)

    # Can't specify to_unit for bool
    with np.testing.assert_raises(ValueError):
        b, safe = RowData({"field": "bool", "to_unit": "rad"}, config, bool)

    # Read the str column
    s, safe = RowData({"field": "str"}, config, str)
    assert safe == True
    assert s == chr(ord("a") + idx)

    # Can't specify to_unit for str
    with np.testing.assert_raises(ValueError):
        s, safe = RowData({"field": "str", "to_unit": "rad"}, config, str)

    # Read the length column
    l, safe = RowData({"field": "length"}, config, float)
    assert safe == True
    assert l == idx

    # Read the length column and convert its units
    l, safe = RowData({"field": "length", "to_unit": "cm"}, config, float)
    assert safe == True
    assert l == idx * 100

    # Read the structured array subfield with units
    dx, safe1 = RowData({"field": "shift", "subfield": "dx"}, config, float)
    dy, safe2 = RowData({"field": "shift", "subfield": "dy"}, config, float)
    dz, safe3 = RowData({"field": "shift", "subfield": "dz"}, config, float)
    assert safe1 == safe2 == safe3 == True
    np.testing.assert_allclose(dx, idx, rtol=0, atol=1e-15)
    np.testing.assert_allclose(dy, (idx + 1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(dz, (idx + 2), rtol=0, atol=1e-15)

    # Read the full structured array with units
    shift, safe = RowData({"field": "shift"}, config, list)
    assert safe == True
    assert len(shift) == 3
    np.testing.assert_allclose(shift[0], idx, rtol=0, atol=1e-15)
    np.testing.assert_allclose(shift[1], (idx + 1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(shift[2], (idx + 2), rtol=0, atol=1e-15)

    # Read the full structured array with units and convert to cm
    shift, safe = RowData({"field": "shift", "to_unit": "cm"}, config, list)
    assert safe == True
    assert len(shift) == 3
    np.testing.assert_allclose(shift[0], idx * 100, rtol=0, atol=1e-15)
    np.testing.assert_allclose(shift[1], (idx + 1) * 100, rtol=0, atol=1e-15)
    np.testing.assert_allclose(shift[2], (idx + 2) * 100, rtol=0, atol=1e-15)


def test_table_row():
    qtable = create_table()
    regular_table = Table(qtable)  # Regular Table (not QTable)
    assert not isinstance(regular_table, QTable)

    # Works for both QTable and Table
    for table in [regular_table, qtable]:
        with TemporaryDirectory() as tmpdir:
            for ext in [".fits", ".ecsv", ".parq"]:
                file_name = tmpdir + "/table_row" + ext
                table.write(file_name, overwrite=True)

                config = {
                    "input": {
                        "table_row": {
                            "file_name": file_name,
                            "keys": ["idx"],
                            "values": [0],
                        },
                    },
                }

                # Check single indexing during table load
                for idx in range(4):
                    config["input"]["table_row"]["values"] = [idx]
                    galsim.config.RemoveCurrent(config["input"]["table_row"])
                    galsim.config.ProcessInput(config)
                    check_row_data(config, idx)

                # Check multi indexing during table load
                for idx0 in range(2):
                    for idx1 in range(2):
                        config["input"]["table_row"]["keys"] = ["idx0", "idx1"]
                        config["input"]["table_row"]["values"] = [idx0, idx1]
                        galsim.config.RemoveCurrent(config["input"]["table_row"])
                        galsim.config.ProcessInput(config)
                        idx = idx0 * 2 + idx1
                        check_row_data(config, idx)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
