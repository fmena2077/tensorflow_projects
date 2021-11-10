# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 19:23:49 2021
Testing dask


@author: francisco mena
"""
import os
import pandas as pd
os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\CNN")


#%%
import dask.dataframe as dd

df = dd.read_csv('datasets/housing/*train*.csv')
df

#%%

train_mean = df.mean(axis = 0).compute()
train_std = df.std(axis = 0).compute()

data = pd.DataFrame({'mean': train_mean, 'std': train_std})
data.drop('MedianHouseValue', axis = 1).to_csv('datasets/housing/stats_trainset.csv')

train_std.T.values
