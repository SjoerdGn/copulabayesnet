# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:31:10 2020

@author: sgnodde
"""


# In this file, a k-fold test with a Bayesian network is done


# Load the package
from copulabayesnet import bncopula as bc
import numpy as np
from copulabayesnet.data_preprocessor import CorrMatrix 


# Load dataset
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True) 
X = X.T # bncopula requires separate rows with the variables

# Let's only use the actual continuous variables
X = np.delete(X, (1), axis = 0)
data = np.vstack([X, y])


#%% Make a file 
import pandas as pd

# some fake titles
titles = 'abcdefghkt'

datadict = {}
for i in range(len(data)):
    datadict[titles[i]] = data[i]
datadf = pd.DataFrame(datadict)

# set to true for creating a file    
if False:
    datadf.to_csv("../../data/copulabayesnet/diabetes.csv", index = False)
    
# Now make a matrix in Uninet
# -> make sure to add the variables in the correct order to the model
    
#%% Do the testing

# Load the matrix
cm = CorrMatrix("example_matrix.txt")

# Create a predict object
pred = bc.Predict(data, [9], R = cm.R)

# 5-fold test with the Gaussian mixture model (mixed gaussians  = 'mg')
nses, kges = pred.k_fold_bn(fit_func = 'mg', n = 500, numpars = 3, k = 5)

print()
print("The NSEs are: ", nses)
print()
print("The KGEs are [kge, rho, alpha, beta]: ", kges)
