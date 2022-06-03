import os
import pandas as pd

def read_xpert(path):
  df = pd.read_csv(path)
  df['patient_id'] = df.Path.apply(lambda x: x.split('/')[2])
  return df

def write_csv(fname, df, root):
  fpath = os.path.join(root, 'CheXpert-v1.0-small', fname)
  with open(fpath, 'w') as f:
    df.to_csv(f, index=False)
  print(f'Wrote to {fpath}')
