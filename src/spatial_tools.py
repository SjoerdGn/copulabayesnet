# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:21:22 2019

@author: GNOS
"""
import numpy as np
    
def inv_dist_weight(distances, b):
    """Inverse distance weight
    
    Parameters
    ----------
    distances : numpy.array of floats
        Distances to point of interest    
    
    b : float
        The parameter of the inverse distance weight. The higher, the 
        higher the influence of closeby stations.
    
    Returns
    -------
    lambdas : numpy.array of floats
        The lambda parameters of the stations
    """

    lambdas = 1/distances**b / np.sum(1/distances**b)
    return lambdas