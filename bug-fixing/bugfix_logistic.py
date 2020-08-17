# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:52:25 2020

@author: sgnodde
"""

from copulabayesnet import bncopula as bc
from copulabayesnet.data_preprocessor import CorrMatrix 
from copulabayesnet import cop_plot as cp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


data_folder = r"../../data/copulabayesnet/CNNpred"

directory = os.fsencode(data_folder)
count = True
for file in os.listdir(directory):
     
    filename = os.fsdecode(file)
    singledata = pd.read_csv(data_folder+'/'+filename)
    name = filename[10:13]
    
    if count:
        data = pd.DataFrame(singledata['Date'])
        count = False    
             
    data[name] = singledata['Close']
data.index = data['Date']
data.index = pd.to_datetime(data.index)
data = data.drop(columns = ['Date'])

pd_data = data.pct_change()
pd_data = pd_data.drop(pd_data.index[0], axis=0)
pd_data = pd_data.rename(columns = {'S&P':'SP'})


cm = CorrMatrix("../examples/example_matrix_stock.txt")
pred = bc.Predict(pd_data.values.T, [0], R = cm.R)
res1 = pred.bn(fit_func = 'logistic', n = 500, numpars = 4, conf_int = 0.9)
print("NSE = ",pred.nse())