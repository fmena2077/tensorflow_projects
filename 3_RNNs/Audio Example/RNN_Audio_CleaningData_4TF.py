# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 18:31:51 2021

@author: franc
"""
import os
os.chdir('C:\\Users\\franc\\projects\\letstf2gpu\\3_ RNNs')

import pandas as pd
import numpy as np

np.random.seed(42)


#%% Clearning

df = pd.read_csv('Paper3_WebData_Final.csv')
print(df.shape)


A = df.head(1000)

datacols = [x for x in df.columns if ('L' in x)|('R' in x)]

df.replace({'**':np.nan}, inplace = True)

df[datacols].isna().sum()

idxwithna = df.loc[df[datacols].isna().any(axis = 1)].index
df.drop(idxwithna, inplace = True)

print(df.shape)


df[datacols] = df[datacols].astype('float32')

#%%

## Left and right are independent

df.dtypes
colsR = [x for x in df.columns if 'R' in x]
colsL = [x for x in df.columns if 'L' in x]
colsX = ['L2k', 'L4k', 'L6k']

data = pd.concat([df[colsL], pd.DataFrame(df[colsR].values, columns = df[colsL].columns) ], axis = 0)

data = pd.concat([ data[colsX].rename(columns={'L2k':'X2k', 'L4k':'X4k', 'L6k':'X6k'}), data], axis = 1)

data.columns = [x.replace('L','Y') for x in data.columns]

print(df.shape)
print(data.shape)

data.min()
data.max()

# Export results

data.to_csv('RNN_Audio_CleanData_4TF.csv')

#%% Export to several files to use later in tf.data

def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    audio_dir = os.path.join("datasets", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path_format = os.path.join(audio_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths


save_to_multiple_csv_files(data[:3000000].values, 'train')
save_to_multiple_csv_files(data[3000000:3500000].values, 'val')
save_to_multiple_csv_files(data[3500000:].values, 'test')

#%% Stats to later normalize data

data[:3000000].mean().values
data[:3000000].std().values

x_mean = [9.60387 ,  8.546144, 10.668388, 15.534396, 20.121899, 22.831743, 22.681715           
        ] #obtained previously from the dataset
x_std = [7.1741357,  7.8720765, 11.261989 , 16.120962 , 18.510359 ,18.409332 , 23.607475           
       ] #obtained previously from the dataset

