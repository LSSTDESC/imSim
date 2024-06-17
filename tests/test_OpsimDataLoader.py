from pathlib import Path
import numpy as np
from imsim import OpsimDataLoader

DATA_DIR = Path(__file__).parent / "data"


def test_seqnum_calculation():
    """Test the values of seqnum computed by OpsimDataLoader from
    an opsim db file."""
    opsim_db_file = str(DATA_DIR / "opsim_db_seqnum_test_data.db")
    visits = sorted(np.random.choice(range(2173), 10, replace=False))
    for visit in visits:
        opsim_data = OpsimDataLoader(opsim_db_file, visit=visit)
        # The seqnum_ref column was added to an existing opsim db
        # file.  It contains the sequence number computed by hand
        # based on the definition of DAYOBS (See appendix A of
        # https://docushare.lsst.org/docushare/dsweb/Get/LSE-400).
        # The test data file contains an excerpt covering two days of
        # observations.
        assert opsim_data.meta["seqnum"] == opsim_data.meta["seqnum_ref"]


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == "test_" and callable(v)]
    for testfn in testfns:
        testfn()
