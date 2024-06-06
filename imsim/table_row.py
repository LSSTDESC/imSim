import astropy.units as u
import galsim
from astropy.table import QTable
from galsim.config import (
    GetAllParams,
    GetInputObj,
    InputLoader,
    RegisterInputType,
    RegisterValueType,
)


class TableRow:
    """Class to extract one row from an astropy QTable and make it available to the
    galsim config layer.

    Parameters
    ----------
    file_name: str
    keys: list
        Column names to use as keys.
    values: list
        Values to match in the key columns.
    """

    _req_params = {
        "file_name": str,
        "keys": list,
        "values": list,
    }

    def __init__(self, file_name, keys, values):
        self.file_name = file_name
        self.keys = keys
        self.values = values
        self.data = QTable.read(file_name)

        for key, value in zip(keys, values):
            self.data = self.data[self.data[key] == value]

        if len(self.data) == 0:
            raise KeyError("No rows found with keys = %s, values = %s" % (keys, values))
        if len(self.data) > 1:
            raise KeyError(
                "Multiple rows found with keys = %s, values = %s" % (keys, values)
            )

    def get(self, field, value_type, from_unit=None, to_unit=None, subfield=None):
        """Get a value from the table row.

        Parameters
        ----------
        field: str
            The name of the column to extract.
        value_type: [float, int, bool, str, galsim.Angle, list]
            The type to convert the value to.
        from_unit: str, optional
            The units of the value in the table.  If the table column already has a
            unit, then this must match that unit or be omitted.
        to_unit: str, optional
            The units to convert the value to.  Only allowed if value_type is one of
            float, int, or list.  Use of this parameter requires that the original units
            of the column are inferrable from the table itself or from from_unit.
        subfield: str, optional
            The name of a subfield to extract from a structured array column.  If
            ommitted, and field refers to a structured array, the entire array is still
            readable as a list value type.

        Returns
        -------
        value: value_type
            The value from the table.
        """
        data = self.data[field]
        if subfield is not None:
            data = data[subfield]

        # See if data already has a unit, if not, add it.
        if data.unit is None:
            if from_unit is not None:
                data = data * getattr(u, from_unit)
        else:
            if from_unit is not None:
                if data.unit != getattr(u, from_unit):
                    raise ValueError(
                        f"from_unit = {from_unit} specified, but field {field} already "
                        f"has units of {data.unit}."
                    )

        # Angles are special
        if value_type == galsim.Angle:
            if to_unit is not None:
                raise ValueError("to_unit is not allowed for Angle types.")

            return float(data.to_value(u.rad)[0]) * galsim.radians

        # For non-angles, we leverage astropy units.
        if to_unit is not None:
            to_unit = getattr(u, to_unit)
        if value_type == list:
            if to_unit is None:
                out = data.value[0].tolist()
            else:
                out = data.to_value(to_unit)[0].tolist()
            # If we had a structured array, `out`` is still a tuple here, so
            # use an extra list() here to finish the cast.
            return list(out)

        # We have to be careful with strings, as using .value on the table datum will
        # convert to bytes, which is not what we want.
        if value_type == str:
            if to_unit is not None:
                raise ValueError("to_unit is not allowed for str types.")
            return str(data[0])

        # No units allowed for bool
        if value_type == bool and to_unit is not None:
            raise ValueError("to_unit is not allowed for bool types.")

        if to_unit is None:
            return value_type(data.value[0])
        else:
            return value_type(data.to_value(to_unit)[0])


def RowData(config, base, value_type):
    row = GetInputObj("table_row", config, base, "table_row")
    req = {"field": str}
    opt = {"from_unit": str, "to_unit": str, "subfield": str}
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt)
    field = kwargs["field"]
    from_unit = kwargs.get("from_unit", None)
    to_unit = kwargs.get("to_unit", None)
    subfield = kwargs.get("subfield", None)
    val = row.get(field, value_type, from_unit, to_unit, subfield)
    return val, safe


RegisterInputType(input_type="table_row", loader=InputLoader(TableRow, file_scope=True))
RegisterValueType(
    type_name="RowData",
    gen_func=RowData,
    valid_types=[float, int, bool, str, galsim.Angle, list],
    input_type="table_row",
)
