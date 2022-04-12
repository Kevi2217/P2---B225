from pathlib import Path

import numpy as np
import pandas as pd

base_dir = Path('C:\\Users\\MikkelNørgaard\\Downloads\\')

file = base_dir / '.csv'

df_a = pd.read_csv(fp_a).set_index(idx_col_a)