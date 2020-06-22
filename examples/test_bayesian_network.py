# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:31:10 2020

@author: sgnodde
"""

# Load the package
from copulabayesnet import bncopula as bc
import matplotlib.pyplot as plt
import numpy as np


# Load dataset
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True) 
X = X.T # bncopula requires rows with the data

# Let's only use the actual continuous variables
X = np.delete(X, (1), axis = 0)
X = np.vstack([X, y])

titles = 'abcdefghkt'
#%% Make a file 
import pandas as pd

datadict = {}
for i in range(len(X)):
    datadict[titles[i]] = X[i]
datadf = pd.DataFrame(datadict)

    
if False:
    datadf.to_csv("../../data/copulabayesnet/diabetes.csv", index = False)
    
    
#%%




