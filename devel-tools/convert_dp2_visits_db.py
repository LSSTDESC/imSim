"""
Script to add the numExposures column the dp2_visits.db file provided
by Rubin to describe DP2 visits.  This will convert that file to a
format consistent with the existing opsim db files that imSim reads.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

db_file = "dp2_visits.db"
assert os.path.isfile(db_file)
with sqlite3.connect(db_file) as con:
    df0 = pd.read_sql("select * from observations", con)

df0['numExposures'] = np.array(df0['nexp'], dtype=int)

outfile = "dp2_visits_fixed.db"
assert not os.path.isfile(outfile)
with sqlite3.connect(outfile) as con:
    df0.to_sql("observations", con, index=False)
