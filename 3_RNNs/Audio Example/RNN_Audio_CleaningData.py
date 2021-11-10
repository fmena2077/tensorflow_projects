# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 18:31:51 2021

@author: francisco mena
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

data = pd.concat([df[colsL], pd.DataFrame(df[colsR].values, columns = df[colsL].columns) ], axis = 0)

print(df.shape)
print(data.shape)

data.min()
data.max()

# Export results

data.to_csv('RNN_Audio_CleanData.csv')
