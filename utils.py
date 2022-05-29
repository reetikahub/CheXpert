import os
import pandas as pd

def write_csv(fname, df, root):
  fpath = os.path.join(root, 'CheXpert-v1.0-small', fname)
  with open(fpath, 'w') as f:
    df.to_csv(f, index=False)
  print(f'Wrote to {fpath}')

def read_csv():
  print('foo')