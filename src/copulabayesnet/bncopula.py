# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:26:59 2019

@author: Sjoerd Gnodde


Required modules: statsmodels, pycopula, scipy, numpy


Note: change d = copula.getDimension() to d = copula.dimension() and 
Sigma = copula.getCovariance() to Sigma = copula.get_corr() in simulation 
in the pycopula module
     
    
    
A small part of the code is based upon https://github.com/blent-ai/pycopula
     

Goodness-of-fit modules
=======================


.. autoclass:: Gof1d
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: Gof2d
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: GofNd
    :members:
    :undoc-members:
    :show-inheritance:

Copulas modules
===============

.. autoclass:: Copula2d
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: CopulaNd
    :members:
    :undoc-members:
    :show-inheritance:


Multivariate normal method
==========================

.. autoclass:: MultVarNorm
    :members:
    :undoc-members:
    :show-inheritance:

Make a prediction
=================

.. autoclass:: Predict
    :members:
    :undoc-members:
    :show-inheritance:
"""
from scipy import interpolate
from mpl_toolkits.mplot3d.axes3d import Axes3D

import scipy as sp  
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from pycopula.copula import GaussianCopula
from pycopula.simulation import simulate
import time
from skgof import ks_test, cvm_test, ad_test

from scipy.stats import norm, uniform, gumbel_r, gumbel_l, shapiro, normaltest, rankdata
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

from pycopula import math_misc, estimation
from pycopula.copula import ArchimedeanCopula 


plt.rc('figure', titlesize=20) 

import hydroeval as he

from sklearn.model_selection import train_test_split

from copulabayesnet import cop_plot as cp

from scipy.stats import multivariate_normal as mvn
from scipy.linalg import cholesky
from scipy.optimize import curve_fit

        
class Gof1d:
    
    """
    Goodness of fit methods for one parameter. Pass either x, x and y or copula
    
    ...
    
    
    Attributes
    ----------
    x : numpy.array or list (optioanl)
        data for x-axis
        
    y : numpy.array or list (optional)
        data for y-axis. Optional because 1d tests only need one dataset.
    
    copula : copula_2d object 
    
    Methods
    -------
    sw_1d
        Shapiro-Wilk test
    
    dag_1d
        D'Agostino test
    
    ad_1d
        Anderson-Darling test
    
    ks_1d
        Kolmogorov-Smirnov test
    
    cvm_1d
        One-dimensional Cramer-von Mises test
        
    auto_corr_test
        Test the autocorrelation of the parameters
    
    distr_func
        Get distribution function from string and data
    
    handle_1d_data
        Handle the 1d data, either from axis or copula.
    
    """
    
    
    def __init__(self, data=None, copula=None):
        
        if data is not None:
            self.data = data
            
        if copula is not None:
            self.x = copula.x
            self.y = copula.y
        
    def sw_1d(self, data=None, axis=None):
        """ Shapiro-Wilk test (https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)
        Test wheter data is normally distributed (Gaussian)
        
        Assumptions:
    
            Observations in each sample are independent and identically distributed (iid).
        
        Interpretation:

            H0: the sample has a Gaussian distribution.
            H1: the sample does not have a Gaussian distribution.

        
        Parameters
        ----------
        data : numpy.array or list (optional)
            The data to test. 
            
        axis : str
            For which axis the data has to be tested. Can be
            either "x" or "y".
            
        Either data or axis should be given.
        
        Returns
        -------
        stat : float
            Shapiro-Wilk statistic
        
        p : float
            p-value
            
        """
        #data
        data = self.handle_1d_data(data, axis)
        
        stat, p = shapiro(data)
        return stat, p
    
    def dag_1d(self, data=None, axis=None):
        """ D'Agostino K^2 normality test (https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)
        
        Assumptions:
    
            Observations in each sample are independent and identically distributed (iid).
        
        Interpretation:

            H0: the sample has a Gaussian distribution.
            H1: the sample does not have a Gaussian distribution.
            
                Parameters
        ----------
        data : numpy.array or list (optional)
            The data to test. 
            
        axis : str
            For which axis the data has to be tested. Can be
            either "x" or "y".
            
        Either data or axis should be given.
        
        Returns
        -------
        stat : float
            D'Agostino statistic
        
        p : float
            p-value
        
        """
        #data
        data = self.handle_1d_data(data, axis)
        
        stat, p = normaltest(data)
        return stat, p
    
    def ad_1d(self, distr, data=None, axis=None):
        """ Anderson-Darling test (https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)
        
        compare to scipy.stats.anderson()
            
        Parameters
        ----------
        distr : str
            The distribution to test against. Should be one of
            - norm
            - gumbel_r
            - gumbel_l
            
        data : numpy.array or list (optional)
            The data to test. 
            
        axis : str
            For which axis the data has to be tested. Can be
            either "x" or "y".
            
        Either data or axis should be given.
        
        Returns
        -------
        GofResult object
            with the AD statistic and the p-value.
        
        """
        
        #data
        data = self.handle_1d_data(data, axis)
        
        #distr
        distr_f = self.distr_func(distr,data)
            
        return ad_test(data, distr_f)
    
    def ks_1d(self, distr, data=None, axis=None):
        """ Kolmogorov-Smirnov supremum statistic (https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
        
        Parameters
        ----------
        distr : str
            The distribution to test against. Should be one of
            - norm
            - uniform
            - gumbel_r
            - gumbel_l
        
        data : numpy.array or list (optional)
            The data to test. 
            
        axis : str
            For which axis the data has to be tested. Can be
            either "x" or "y".
            
        Either data or axis should be given.
        
        Returns
        -------
        GofResult object
            with the KS statistic and the p-value.
            
        """
        #data
        data = self.handle_1d_data(data, axis)
            
        #distr
        distr_f = self.distr_func(distr,data)
            
        return ks_test(data, distr_f)
        
    def cvm_1d(self, distr, data=None, axis=None):
        """ Cramer-von Mises test (https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion)
        
        Parameters
        ----------
        distr : str
            The distribution to test against. Should be one of
            - norm
            - uniform
            - gumbel_r
            - gumbel_l
        
        data : numpy.array or list (optional)
            The data to test. 
            
        axis : str
            For which axis the data has to be tested. Can be
            either "x" or "y".
            
        Either data or axis should be given.
        
        Returns
        -------
        GofResult object
            with the CvM statistic and the p-value.
            
        """
        #data
        data = self.handle_1d_data(data, axis)
        
        #distr
        distr_f = self.distr_func(distr,data)
            
        return cvm_test(data, distr_f)
    
    def auto_corr_test(self, 
                       data=None, 
                       axis=None, 
                       lag = 1, 
                       return_all = False,
                       plot_figure = False,
                       margin = 0.15,
                       data_name = "data",
                       save_fig = False,
                       save_path = "autocorrelation.png"):
        """ Test autocorrelation: one indicator
        that the samples are independent and identically distributed (iid).

            
                Parameters
        ----------
        data : numpy.array or list (optional)
            The data to test. 
            
        axis : str
            For which axis the data has to be tested. Can be
            either "x" or "y".
            
        Either data or axis should be given.
        
        lag : int
            The steps be Default value: 1
            
        plot_figure : bool (optional)
            If True, plot the figure
        
        margin : float (optional)
            The margin of the greyed out area in the plot.
            Default value: 0.15
        
        data_name : str (optional)
            The name of the data that is being plotted such that 
            it gets a nice title
            
        save_fig : bool (optional)
            Save the figure to a file
            
        save_path : str (optional)
            Full or relative path to save the figure. 
            Default value: "cdf_2d_test.png"
        
        
        Returns
        -------
        auto_corr : float or np.array of floats
            auto correlation
        
        """
        #data
        try:
            data = self.handle_1d_data(data, axis)
        except:
            data = self.data
        
        auto_corr = acf(data, unbiased = True, nlags=lag)
        
        if plot_figure:
   
            fig = plt.figure(figsize = (12,8))
            ax = fig.add_subplot(111)
            ax.bar(np.linspace(0,lag,lag+1),auto_corr)
            
            year = 0
            while year <= lag:
                ax.axvline(year, label = "One year", linestyle = (0, (1, 5)), color = 'firebrick', linewidth = 0.9)
                year = year+12
            
            shade_1 = [-margin, -margin]
            shade_2 = [margin, margin]
            ax.fill_between([-1, lag+1], shade_1, shade_2, color = 'lightgrey')
            ax.axhline(0, color = 'black', linewidth = 0.8)
            ax.set_xlim(-1, lag+1)
            ax.set_xticks(np.arange(0, lag+1, 6))
            ax.set_xlabel("Lag in months")
            ax.set_ylabel("Autocorrelation")
            ax.set_title("Autocorrelation of "+data_name)
            ax.set_ylim(-0.55, 1.1)
        
            if save_fig:
                plt.savefig(save_path, dpi=200)
        
        if not return_all:
            auto_corr = auto_corr[-1]
        return auto_corr
    
    def handle_1d_data(self, data, axis):
        
        """Handle incoming data
        """
        #data
        if data is not None:
            data = data
        elif axis.lower() == "x":
            data = self.x
        elif axis.lower() == "y":
            data = self.y
        else:
            data = self.x
        return data
    
    def distr_func(self, distr, data):
        
        """Make a distribution function object from string
        """
        #distr
        if distr == "norm":
            mu, sig = norm.fit(data)
            distr_f = norm(mu, sig)
            
        elif distr == "uniform":
            begin, len_unif = uniform.fit(data)
            distr_f = uniform(begin, len_unif)
        
        elif distr == "gumbel_r": 
            loc, scale = gumbel_r.fit(data)
            distr_f = gumbel_r(loc, scale)
            
        elif distr == "gumbel_l": 
            loc, scale = gumbel_l.fit(data)
            distr_f = gumbel_l(loc, scale)
        
        else:
            print("Distr should be either norm, uniform, gumbel_r or gumbel_l")
            
        return distr_f


class Gof2d:   
    """
    Goodness of fit methods for the 2-dimensional copula. Pass either x and y or copula
    
    ...
    
    
    Attributes
    ----------
    x : numpy.array or list (optioanl)
        data for x-axis
        
    y : numpy.array or list (optional)
        data for y-axis. Optional because 1d tests only need one dataset.
        
    copula : copula_2d object 
    
    Methods
    -------          
    quadrant_rho
        Pearson's correlation for all of the quadrants
    
    quadrants
        Get quadrants from 2d data 
        
    cdf_2d_test
        Test the empirical copula vs the theoretical copula.
        
        
    """

    def __init__(self, x=None, y=None, copula=None):
        
        if x is not None:
            self.x = x
        
        if y is not None:
            self.copula = Copula2d(x,y)
            self.y = y
            
        if copula is not None:
            self.copula = copula
            self.x = copula.x
            self.y = copula.y
            
        self.copula_fitted = False
        
        
    def cvm_empemp(self):
        """ Cramer-von Mises test for two empirical distributions:
            Aren't they actually the same?
            
        Returns
        -------
        T : float
            
        
        """
        x = self.x
        y = self.y
        
        N = len(x)
        M = len(y)
        
        rank_x = rankdata(x)
        rank_y = rankdata(y)
        U = N*np.sum([(rank_x[i]-i)**2 for i in range(N)])+M*np.sum([(rank_y[i]-i)**2 for i in range(M)])


        T = U/(N*M*(N+M))-(4*M*N-1)/(6*(M+N))
        return T
    
    def quadrant_rho_complex(self, distrib, N = 50,
                     n="Default", 
                     plot_bar = False, 
                     save_fig = False, 
                     save_path = "quad_plot.png",
                     plot_scatter_=False,
                     data_names="",
                     **kwargs):
        """ Test what the standard deviation per quadrant is, and what the
        theoretical one would be.
        
        Parameters
        ----------
        distrib : str
            Distribution
            For now, either:
                - norm
                
        n : int (optional)
            Number of samples in theoretical distribution
            
        plot_bar : bool (optional)
            Default value: False
            Plot the rhos
        
        save_fig : bool (optional)
            Default value: False
            Save the plot when true
        
        save_path : str (optional)
            Default: "quad_plot.png"
            The location where to save the file. 
        
        plot_scatter : bool (optional)
            Default: False
            Make a scatter plot
        
        **kwargs
            To be passed to plt.figure(**kwargs)
        
        
        Returns
        -------
        param_rho
            dictionary with Pearson coefficients in quadrants
            nw, ne, se, sw of the input values 
        theor_rho
            the same for the sampled theoretical distribution       
        
        Optional: plot of the array mentioned above
         
        """
        x = np.array(self.x)
        y = np.array(self.y)
        copula = self.copula
        unif_x = copula.val2unif(x, "x")
        unif_y = copula.val2unif(y, "y")
        
        
        if distrib == "norm": 
            norm_x_unfiltered = norm.ppf(unif_x)
            norm_y_unfiltered = norm.ppf(unif_y)
        
        else:
            print("For now, only normal distributions are supported")
        
        
        
        mask_x = ~np.isinf(norm_x_unfiltered)
        x_norm = norm_x_unfiltered[mask_x]
        
        
        mask_y = ~np.isinf(norm_y_unfiltered)
        y_norm = norm_y_unfiltered[mask_y]
        
        param_quad = self.quadrants(x_norm, y_norm)
        copula = self.copula
              
        
        if not self.copula_fitted:
            copula.fit()
            
        
        if n == "Default":
            n = len(x)
        
        
        theor_vals_total = np.zeros((4, N))
        
        for i in range(N):
            # Sample
            copula.sample(n=n)   
            theor_vals = [[i[0] for i in copula.sample_values], 
                          [i[1] for i in copula.sample_values]]
    
    
            theor_norm = norm.ppf(theor_vals)
            theor_quad = self.quadrants(np.array(theor_norm[0]),
                                       np.array(theor_norm[1]))
            #print(theor_quad.keys())
            for j in range(4):
                theor_vals_total[j,i] = np.corrcoef(theor_quad[list(theor_quad.keys())[j]][0],
                     theor_quad[list(theor_quad.keys())[j]][1])[0,1]
                
        #print(theor_quad)

        param_rho = {}
        
        
        for quad in param_quad:
            param_rho[quad] = np.corrcoef(param_quad[quad][0],
                     param_quad[quad][1])[0,1]
        
        theor_rho = {}
        for i in range(4):
            theor_rho[list(theor_quad.keys())[i]] = np.mean(theor_vals_total[i])
            
            
        def plot_vals(param_rho, 
                      theor_rho,
                      distrib,
                      save_fig=False, 
                      save_path="quadrant_test.png", 
                      **kwargs):
            plt.figure(**kwargs)
            plt.bar([1,2,3,4], theor_rho.values(), width = 0.3, label="Theoretical samples")
            plt.bar(np.array([1,2,3,4])+0.3, param_rho.values(), width = 0.3, label = "Parameter samples")
            plt.xticks(np.array([1,2,3,4])+0.15,[*theor_rho])
            plt.legend()
            plt.ylabel('Pearsons correlation coefficient')
            plt.title(f"Quadrant correlation test: {distrib} of {data_names}")
            plt.axhline(0, color = 'black')
            plt.grid(axis = 'y')

            if save_fig:
                plt.savefig(save_path)
        
        if plot_bar:
            plot_vals(param_rho, 
                      theor_rho, 
                      distrib,
                      save_fig=save_fig, 
                      save_path=save_path, 
                      **kwargs)
            
        def plot_scatter():
            plt.figure(**kwargs)
            print(np.array([x_norm, y_norm]))
            cp.scatter_hist(np.array([x_norm, y_norm]))
            
        print(plot_scatter_)
        if plot_scatter_:
            print("Test")
            plot_scatter()
            
        return param_rho, theor_rho
            
    def quadrant_rho(self, distrib,
                     n="Default", 
                     family = 'gumbel',
                     plot_bar = False, 
                     save_fig = False, 
                     save_path = "quad_plot.png",
                     plot_scatter_=False,
                     data_names="",
                     **kwargs):
        """ Test what the standard deviation per quadrant is, and what the
        theoretical one would be.
        
        Parameters
        ----------
        distrib : str
            Distribution
            For now, either:
                - norm
                
        n : int (optional)
            Number of samples in theoretical distribution
            
        plot_bar : bool (optional)
            Default value: False
            Plot the rhos
        
        save_fig : bool (optional)
            Default value: False
            Save the plot when true
        
        save_path : str (optional)
            Default: "quad_plot.png"
            The location where to save the file. 
        
        plot_scatter : bool (optional)
            Default: False
            Make a scatter plot
        
        **kwargs
            To be passed to plt.figure(**kwargs)
        
        
        Returns
        -------
        param_rho
            dictionary with Pearson coefficients in quadrants
            nw, ne, se, sw of the input values 
        theor_rho
            the same for the sampled theoretical distribution       
        
        Optional: plot of the array mentioned above
         
        """
        x = np.array(self.x)
        y = np.array(self.y)
        copula = self.copula
        unif_x = copula.val2unif(x, "x")
        unif_y = copula.val2unif(y, "y")
        
        if distrib == "norm": 
            norm_x_unfiltered = norm.ppf(unif_x)
            norm_y_unfiltered = norm.ppf(unif_y)
        
        else:
            print("For now, only normal distributions are supported")
        
        
        
        mask_x = ~np.isinf(norm_x_unfiltered)
        x_norm = norm_x_unfiltered[mask_x]
        
        
        mask_y = ~np.isinf(norm_y_unfiltered)
        y_norm = norm_y_unfiltered[mask_y]
        
        #plt.figure()
        #plt.scatter(x_norm, y_norm)
        
        
        param_quad = self.quadrants(x_norm, y_norm)
        copula = self.copula
              
        
        if not self.copula_fitted:
            copula.fit()
            
        
        if n == "Default":
            n = len(x)
        
        if family.lower() == 'gaussian':
            theor_cop = GaussianCopula(dim = 2)
            
        else:
            theor_cop = ArchimedeanCopula(family=family, dim=2)

        theor_cop.fit(np.array([x,y]).T)
            
        smp_pre = simulate(theor_cop, n) 
        theor_vals = np.array([[smp_pre[i][j] for i in range(n)] for j in range(2)]) 
        
        inv_theor_vals = np.array(theor_vals)
        
        #print(inv_theor_vals)
        #plt.figure()
        #plt.scatter(inv_theor_vals[0], inv_theor_vals[1])

        theor_norm = norm.ppf(theor_vals)
        theor_quad = self.quadrants(np.array(theor_norm[0]),
                                   np.array(theor_norm[1]))

        param_rho = {}
        theor_rho = {}
        
        for quad in param_quad:
            param_rho[quad] = np.corrcoef(param_quad[quad][0],
                     param_quad[quad][1])[0,1]
            
        for quad in theor_quad:
            theor_rho[quad] = np.corrcoef(theor_quad[quad][0],
                     theor_quad[quad][1])[0,1]
            
        def plot_vals(param_rho, 
                      theor_rho,
                      distrib,
                      save_fig=False, 
                      save_path="quadrant_test.png", 
                      **kwargs):
            plt.figure(**kwargs)
            plt.bar([1,2,3,4], theor_rho.values(), width = 0.3, label="Theoretical samples")
            plt.bar(np.array([1,2,3,4])+0.3, param_rho.values(), width = 0.3, label = "Parameter samples")
            plt.xticks(np.array([1,2,3,4])+0.15,[*theor_rho])
            plt.legend()
            plt.ylabel('Pearsons correlation coefficient')
            plt.title(f"Quadrant correlation test: {distrib} of {data_names}")
            plt.axhline(0, color = 'black')
            plt.grid(axis = 'y')

            if save_fig:
                plt.savefig(save_path)
        
        if plot_bar:
            plot_vals(param_rho, 
                      theor_rho, 
                      distrib,
                      save_fig=save_fig, 
                      save_path=save_path, 
                      **kwargs)
            
        def plot_scatter():
            plt.figure(**kwargs)
            #print(np.array([x_norm, y_norm]))
            cp.scatter_hist(np.array([x_norm, y_norm]))
            
        if plot_scatter_:
            plot_scatter()
            
        return param_rho, theor_rho
    
    
    def quadrant_tail_dep(self, distrib="norm" ):
        """ Test what the standard deviation per quadrant is, and what the
        theoretical one would be.
        
        Parameters
        ----------
        distrib : str
            Distribution
            For now, either:
                - norm

        
        Returns
        -------
        param_rho
            dictionary with Pearson coefficients in quadrants
            nw, ne, se, sw of the input values 
        all_rho
            correlation coefficient for all   
        
        Optional: plot of the array mentioned above
         
        """
        x = np.array(self.x)
        y = np.array(self.y)
        copula = self.copula
        unif_x = copula.val2unif(x, "x")
        unif_y = copula.val2unif(y, "y")
        
        if distrib == "norm": 
            norm_x_unfiltered = norm.ppf(unif_x)
            norm_y_unfiltered = norm.ppf(unif_y)
        
        else:
            print("For now, only normal distributions are supported")
        
    
        mask_x = ~np.isinf(norm_x_unfiltered)
        x_norm = norm_x_unfiltered[mask_x]
        
        
        mask_y = ~np.isinf(norm_y_unfiltered)
        y_norm = norm_y_unfiltered[mask_y]
        
   
        param_quad = self.quadrants(x_norm, y_norm)

        param_rho = {}
        
        for quad in param_quad:
            param_rho[quad] = np.corrcoef(param_quad[quad][0],
                     param_quad[quad][1])[0,1]

        overall_rho = np.corrcoef(x_norm, y_norm)[0,1]
        return param_rho, overall_rho
        
        
    def quadrants(self, x, y):
        
        quadrants = {}
        nw_mask = np.logical_and(x <= 0, y >= 0)
        quadrants['nw'] = np.array([x[nw_mask], y[nw_mask]])
        ne_mask = np.logical_and(x > 0, y >= 0)
        quadrants['ne'] = np.array([x[ne_mask], y[ne_mask]])
        se_mask = np.logical_and(x > 0, y < 0)
        quadrants['se'] = np.array([x[se_mask], y[se_mask]])
        sw_mask = np.logical_and(x <= 0, y < 0)
        quadrants['sw'] = np.array([x[sw_mask], y[sw_mask]])
            
        return quadrants
    
    
    def cvm_2d(self, 
                    formula = "RMSD", 
                    plot_figure = False, 
                    plot_3d = False,
                    x_name = "x-axis", 
                    y_name = "y-axis",
                    title = "Difference between theoretical and empirical copula ($C_{emp}-C_{theor}$)",
                    levels = 30,
                    gridlines = True,
                    save_fig = False,
                    save_path = "cdf_2d_test.png"):
            
        """ 2d Cramer-von Mises. Test the 3d (2-variable) theoretical copula vs the empirical one 
        (directly from the data). Calculates the RMSE and is also able to plot 
        the exact difference.
        
        Parameters
        ----------
        formula : str (optional)
            Default value: "RMSD"
            The formula behind the method. Should be either "rmse", 
            "rmsd", "mse", "msd", "genest", "basic", "original", "morales", 
            "morales-napoles" or "times_n"
        
        plot_figure : bool (optional)
            If True, plot the figure
            
        plot_3d : bool (optional)
            Plot the figure in 3d
        
        x_name :  str (optional)
            The name to plot on the x-axis. Default value: "x-axis"
        
        y_name :  str (optional)
            The name to plot on the x-axis. Default value: "y-axis"
            
        title : str (optional)
            The title of the plot. Default value: "Difference between
            theoretical and empirical copula (F_emp-F_theor)" 
        
        levels : int (optional)
            Number of levels in the data. Default value: 30
        
        gridlines : bool (optional)
            Plot gridlines. Default value: True
        
        save_fig : bool (optional)
            Save the figure to a file
            
        save_path : str (optional)
            Full or relative path to save the figure. 
            Default value: "cdf_2d_test.png"
        
        
        Returns
        -------
        statistic : float
            The Cramer-von Mises statistic.
        """
        
        copula = self.copula
        rank_x = rankdata(self.x)
        rank_y = rankdata(self.y)
        
        tot = len(rank_x)
        emppoints = np.array([rank_x, rank_y])/tot
        
        point_val = np.zeros_like(rank_x)
        diff_point_val = np.zeros_like(rank_x)
      
        for i in range(tot):
            nums = np.where(np.logical_and(emppoints[0] <= emppoints[0,i],emppoints[1] <= emppoints[1,i]))
            z_pract = len(nums[0])/tot
            point_val[i] = z_pract
            z_theor = copula.copula.cdf([[emppoints[0,i], emppoints[1,i]]])
            diff = z_pract-z_theor
            diff_point_val[i] = diff
               
        sq_diff_point_val = diff_point_val**2
               
        if formula.lower() in ["rmse", "rmsd"]:
            statistic = np.nanmean(sq_diff_point_val)**0.5
        elif formula.lower() in ["mse", "msd"]:
            statistic = np.nanmean(sq_diff_point_val)
        elif formula.lower() in ["genest", "basic", "original"]: 
            statistic = sq_diff_point_val
        elif formula.lower() in ["morales", "morales-napoles", "times_n"]:
            statistic = len(sq_diff_point_val)*sq_diff_point_val
        else:
            raise ValueError('Parameter formula should be either "rmse", "rmsd", "mse", "msd", "genest", "basic", "original", "morales", "morales-napoles" or "times_n"')
        
        if plot_figure:
            if plot_3d:
                #from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(12,8))
                #ax = fig.add_subplot(111, projection='3d')
                ax = Axes3D(fig)
                plot_title = title+f", Statistic = {round(statistic,4)}"
                x_3d, y_3d = np.meshgrid([0, 0, 1, 1], [0, 1, 0, 1]) 
                ax.plot([0,0,1], [0,1,1], color = 'b', zorder = 0)
                ax.plot([0,1,1],[0,0,1], color = 'b', zorder = 3)
                ax.plot_trisurf(rank_x/tot, rank_y/tot, diff_point_val, cmap="RdBu_r", zorder = 2)
                ax.set_title(plot_title)
                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)
                ax.set_zlabel("Difference between copulas")
                if save_fig:
                    plt.savefig(save_path, dpi=200)
                
            
            else:
                plot_title = title#+f", RMSD = {round(statistic,4)}"
                plt.figure(figsize=(10,8))
                if gridlines:
                    plt.tricontour(rank_x/tot, rank_y/tot, diff_point_val, levels = levels, colors='k',linewidths=0.5)
                trc = plt.tricontourf(rank_x/tot, rank_y/tot, diff_point_val, levels = levels, cmap="RdBu_r")
                plt.title(plot_title,   fontsize = 18, pad = 20)
                plt.xlabel(x_name,   color = '#4d4d4d', fontsize = 14)
                plt.ylabel(y_name,   color = '#4d4d4d', fontsize = 14)
                plt.xticks(  color = '#4d4d4d', fontsize = 14)
                plt.yticks( color = '#4d4d4d', fontsize = 14)
                plt.colorbar(trc)
                plt.grid(which='major', axis='both', color='darkgrey',
                         linestyle='dashdot', linewidth=0.3)
                plt.axis('square')
                plt.tight_layout()
                if save_fig:
                    plt.savefig(save_path, dpi=200)
                    

            
        return statistic
    
    
    def cvm_2d_p(self):
        """
        """
        print("Programme empty: Please use the Nd one for this, for now. It does the same this will do")
    
class GofNd:
    """Goodness of fit tests for the n-dimensional copula
    
    """
    
    def __init__(self, data=None, copula=None):
        
        # get data
        if data is not None:
            self.data = data
        elif copula is not None:
            self.data = copula.data
            
        # get copula
        if copula is not None:
            self.copula = copula
        elif data is not None:
            self.copula = CopulaNd(data)
            self.copula.fit()
        else:
            print("You should either pass data or a copula object")
            
    def cvm_nd(self, formula = "RMSD", copula=None, data=None):
        
        """ nd Cramer-von Mises. Test the theoretical copula vs the empirical one 
        (directly from the data). Calculates the RMSE and is also able to plot 
        the exact difference.
        
        Parameters
        ----------
        formula : str (optional)
            Default value: "RMSD"
            The formula behind the method. Should be either "rmse", 
            "rmsd", "mse", "msd", "genest", "basic", "original", "morales", 
            "morales-napoles" or "times_n"
            
        copula : CopulaNd object (optional)
            Enter copula object manually.
        
        data : dim*n numpy.array
            Enter data manually.
        
        Returns
        -------
        statistic : float
            The Cramer-von Mises statistic.
        """
        
        #HIER WAS IK
        
        if copula is None:
            copula = self.copula
        
        if data is None:
            data = self.data
            
        copula.check_fit()
            
        dim = copula.dim
        
        rank_data = [rankdata(data[i]) for i in range(copula.dim)]
        
        tot = len(rank_data[0])
        emppoints = np.array(rank_data)/tot
        
        point_val = np.zeros(tot)
        diff_point_val = np.zeros(tot)
      
        for i in range(tot):
            test_array = np.array([emppoints[j] <= emppoints[j,i] for j in range(dim)])
            nums = np.where(np.logical_and.reduce(test_array))
            z_pract = len(nums[0])/tot
            point_val[i] = z_pract
            
            z_theor_vals = [[emppoints[j,i] for j in range(dim)]]
            z_theor = copula.big_c(z_theor_vals[0])
            
            diff = z_pract-z_theor
            diff_point_val[i] = diff
               
        sq_diff_point_val = diff_point_val**2
        if formula.lower() in ["rmse", "rmsd"]:
            statistic = np.nanmean(sq_diff_point_val)**0.5
        elif formula.lower() in ["mse", "msd"]:
            statistic = np.nanmean(sq_diff_point_val)
        elif formula.lower() in ["genest", "basic", "original"]: 
            statistic = sq_diff_point_val
        elif formula.lower() in ["morales", "morales-napoles", "times_n"]:
            statistic = len(sq_diff_point_val)*sq_diff_point_val
        else:
            raise ValueError('Parameter formula should be either "rmse", "rmsd", "mse", "msd", "genest", "basic", "original", "morales", "morales-napoles" or "times_n"')
            
        return statistic
    
    def cvm_nd_p(self, N, formula = "RMSD", ignore_messages=False):
        
        """ Calculate the p value for H0: the empirical copula 
        is not a Gaussian copula. 
        
        Parameters
        ----------
        N : int
            The number of times to run the test to get the p-value
            
        formula : str (optional)
            Default value: "RMSD"
            The formula behind the method. Should be either "rmse", 
            "rmsd", "mse", "msd", "genest", "basic", "original", "morales", 
            "morales-napoles" or "times_n"
            
        ignore_messages : bool (optional)
            Default value: False
            When true, only prints time it took but ignores rest of the messages.
        
        Returns
        -------
        p : float
            The p value H0: the empirical copula 
            is not a Gaussian copula.
        
        """
        time0 = time.time()
        base_copula = self.copula
        base_data = self.data
        base_statistic = self.cvm_nd(formula=formula)
        star_statistics = np.zeros(N)
        dim = base_copula.dim
        
        if not ignore_messages:
                print("Base statistic data ", formula, ": ", base_statistic)
        
        n = len(base_data[0])
        
        for i in range(N):
            smp_pre = base_copula.sample(n)
            sampvals = np.array([[smp_pre[i][0][j] for i in range(n)] for j in range(dim)])
            star_cop = CopulaNd(sampvals)
            star_cop.fit(ignore_message=True)
            star_statistic = self.cvm_nd(formula=formula, copula=star_cop, data = sampvals)
            star_statistics[i] = star_statistic
            if not ignore_messages:
                print("Statistic ", formula, ": ", star_statistic)
        
        p = np.sum(star_statistics>base_statistic)/N
        
        
        print(f"Calculations took {time.time()-time0} seconds")
        return p
        
class GofNdv2:
    """Goodness of fit tests for the n-dimensional copula
    
    """
    
    def __init__(self, data=None, copula=None):
        
        # get data
        if data is not None:
            self.data = data
        elif copula is not None:
            self.data = copula.data
            
        # get copula
        if copula is not None:
            self.copula = copula
        elif data is not None:
            pass
        #     self.copula = CopulaNd(data)
        #     self.copula.fit()
        else:
            print("You should either pass data or a copula object")
            
    def cvm_nd(self, formula = "RMSD", copula=None, data=None):
        
        """ nd Cramer-von Mises. Test the theoretical copula vs the empirical one 
        (directly from the data). Calculates the RMSE and is also able to plot 
        the exact difference.
        
        Parameters
        ----------
        formula : str (optional)
            Default value: "RMSD"
            The formula behind the method. Should be either "rmse", 
            "rmsd", "mse", "msd", "genest", "basic", "original", "morales", 
            "morales-napoles" or "times_n"
            
        copula : CopulaNd object (optional)
            Enter copula object manually.
        
        data : dim*n numpy.array
            Enter data manually.
        
        Returns
        -------
        statistic : float
            The Cramer-von Mises statistic.
        """
        
        #HIER WAS IK
        
        if copula is None:
            copula = self.copula
        
        if data is None:
            data = self.data
            
        #copula.check_fit()
            
        dim = copula.dim
        
        rank_data = [rankdata(data[i]) for i in range(copula.dim)]
        
        tot = len(rank_data[0])
        emppoints = np.array(rank_data)/tot
        
        point_val = np.zeros(tot)
        diff_point_val = np.zeros(tot)
      
        for i in range(tot):
            test_array = np.array([emppoints[j] <= emppoints[j,i] for j in range(dim)])
            nums = np.where(np.logical_and.reduce(test_array))
            z_pract = len(nums[0])/tot
            point_val[i] = z_pract
            
            z_theor_vals = [[emppoints[j,i] for j in range(dim)]]
            z_theor = copula.cdf(z_theor_vals[0])#copula.big_c(z_theor_vals[0])
            
            diff = z_pract-z_theor
            diff_point_val[i] = diff
               
        sq_diff_point_val = diff_point_val**2
        if formula.lower() in ["rmse", "rmsd"]:
            statistic = np.nanmean(sq_diff_point_val)**0.5
        elif formula.lower() in ["mse", "msd"]:
            statistic = np.nanmean(sq_diff_point_val)
        elif formula.lower() in ["genest", "basic", "original"]: 
            statistic = sq_diff_point_val
        elif formula.lower() in ["morales", "morales-napoles", "times_n"]:
            statistic = len(sq_diff_point_val)*sq_diff_point_val
        else:
            raise ValueError('Parameter formula should be either "rmse", "rmsd", "mse", "msd", "genest", "basic", "original", "morales", "morales-napoles" or "times_n"')
            
        return statistic
    
    def cvm_nd_p_rest(self, N, copula = None,  formula = "RMSD", ignore_messages=False):
        
        """ Calculate the p value for H0: the empirical copula 
        is not a Gaussian copula. 
        
        Parameters
        ----------
        N : int
            The number of times to run the test to get the p-value
            
        formula : str (optional)
            Default value: "RMSD"
            The formula behind the method. Should be either "rmse", 
            "rmsd", "mse", "msd", "genest", "basic", "original", "morales", 
            "morales-napoles" or "times_n"
            
        ignore_messages : bool (optional)
            Default value: False
            When true, only prints time it took but ignores rest of the messages.
        
        Returns
        -------
        p : float
            The p value H0: the empirical copula 
            is not a Gaussian copula.
        
        """
        time0 = time.time()
        if copula is None:
            print("None")
            copula = self.copula
        base_copula = copula
        base_data = self.data
        base_statistic = self.cvm_nd(formula=formula, copula = copula)
        star_statistics = np.zeros(N)
        dim = base_copula.dim
        
        if not False:
                print("Base statistic data ", formula, ": ", base_statistic)
        
        n = len(base_data[0])
        
        for i in range(N):
            if i%10 == 0:
                print(f"{round(i*100/N)}% - ", end='')
            smp_pre = simulate(base_copula, n)  # Step a #base_copula.sample(n) 
            sampvals = np.array([[smp_pre[i][j] for i in range(n)] for j in range(dim)])  #np.array([[smp_pre[i][0][j] for i in range(n)] for j in range(dim)])
            star_cop = ArchimedeanCopula(family=base_copula.getFamily(), dim=2)#CopulaNd(sampvals.T)
            star_cop.fit(sampvals.T) #step b
            star_statistic = self.cvm_nd(formula=formula, copula=star_cop, data = sampvals) # step c, d
            star_statistics[i] = star_statistic
            if not ignore_messages:
                print("Statistic ", formula, ": ", star_statistic)
        
        p = np.sum(star_statistics>base_statistic)/N
        
        
        print(f"Calculations took {time.time()-time0} seconds")
        return p
        
    def cvm_nd_p_gauss(self, N, copula = None, formula = "RMSD", ignore_messages=False):
        
        """ Calculate the p value for H0: the empirical copula 
        is not a Gaussian copula. 
        
        Parameters
        ----------
        N : int
            The number of times to run the test to get the p-value
            
        formula : str (optional)
            Default value: "RMSD"
            The formula behind the method. Should be either "rmse", 
            "rmsd", "mse", "msd", "genest", "basic", "original", "morales", 
            "morales-napoles" or "times_n"
            
        ignore_messages : bool (optional)
            Default value: False
            When true, only prints time it took but ignores rest of the messages.
        
        Returns
        -------
        p : float
            The p value H0: the empirical copula 
            is not a Gaussian copula.
        
        """
        time0 = time.time()
        base_copula = copula
        base_data = self.data
        base_statistic = self.cvm_nd(formula=formula, copula = copula)
        star_statistics = np.zeros(N)
        dim = base_copula.dim
        
        if not False: #ignore_messages:
                print("Base statistic data ", formula, ": ", base_statistic)
        
        n = len(base_data[0])
        
        for i in range(N):
            if i%10 == 0:
                print(f"{round(i*100/N)}% - ", end='')
            smp_pre = simulate(base_copula, n)
            #print(smp_pre)
            sampvals = np.array([[smp_pre[i][j] for i in range(n)] for j in range(dim)])
            star_cop = GaussianCopula(dim = 2)
            star_cop.fit(sampvals.T, ignore_message=True)
            star_statistic = self.cvm_nd(formula=formula, copula=star_cop, data = sampvals)
            star_statistics[i] = star_statistic
            if not ignore_messages:
                print("Statistic ", formula, ": ", star_statistic)
        
        p = np.sum(star_statistics>base_statistic)/N
        #plt.hist(star_statistics)
        
        print(f"Calculations took {time.time()-time0} seconds")
        return p
    
    
class Copula2d:
    """
    
    An empirical, two-dimensional, Gaussian copula with several useful methods.
    
    ...
    
    
    Attributes
    ----------
    x : numpy.array 
        input data emperical copula
    y : numpy.array
        input data empirical copula
    copula : GaussianCopula object from the module pycopula
        Copula object. Methods can be used
    ecdf_x : ECDF object from the module statsmodels
        The empirical CDF of the x-data. Methods can be used.
    ecdf_y : ECDF object from the module statsmodels
        The empirical CDF of the y-data. Methods can be used.
    x_name : string (optional)
        Manual name of x-axis.
    y_name : string (optional)
        Manual name of y-axis.
    samples : np.array of floats (optional)
        Samples from the copula.    
    
    
    Methods
    -------
    
    axis_from_data_name(name)
        Get the build-in axis name (x or y) from the assigned name

    cond_expected_value(val, cond_axis, samp_range=0.005)
         Calculate the conditional expected value
         (mean of the sampled values) from one 
         parameter to the other

    cond_std(val, cond_axis, samp_range=0.005)
        Calculate the conditional standard deviation from one 
        parameter to the other

    emp_corr(matrix=False)
        Calculate the empirical correlation (Pearson's r)
  
    emp_norm_rank_corr(self, matrix=False)
        Calculate the empirical normal rank correlation

    emp_rank_corr(self, matrix=False)
        Calculate the empirical rank correlation (Spearman's rho)

    fit(self)
        Fit the copula with correlation and data 

    sample(self, n=1000)
        Generate sample values from the copula.

    set_data_names(self, x_name=None, y_name=None)
        Set the names of the data. Useful for documenting in 
        which what is saved

    uncond_expected_value(self, axis)
        Calculate the unconditional expected value
        (mean of the values) of one dataset

    uncond_std(self, axis)
        Calculate the unconditional standard deviation
        of one dataset

    unif2unifs(self, unival, samp_range=0.005)
        From the uniform value to the uniform values of the other parameter

    unif2val(self, unival, ecdf)
        Get the value from a single uniform value

    unifs2vals(self, univals, axis='y')
        Find the mean value from the uniform values
   
     val2unif(self, val, axis)
         Calculate the uniform variant of a parameter value

    """


    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.copula = GaussianCopula()
        self.ecdf_x = ECDF(x)
        self.ecdf_y = ECDF(y)
        self.fitted = False

        
    
    def emp_norm_rank_corr(self, matrix = False):
        """Calculate the empirical normal rank correlation.
        
        If matrix, returns empirical normal rank correlation 
        in matrix form instead of single number. 
        
        
        Parameters
        ----------
        matrix : bool, optional
            return matrix instead of single number.
            
        """
        x = self.x
        y = self.y

        rank_x = sp.stats.rankdata(x)/(len(x)+1)
        rank_y = sp.stats.rankdata(y)/(len(y)+1)
        
        invnorm_x = sp.stats.norm.ppf(rank_x)
        invnorm_y = sp.stats.norm.ppf(rank_y)
        
        # mask invalid data?
        pearson_rank = sp.stats.pearsonr(invnorm_x, invnorm_y)[0]
    
        emp_norm_rank_corr = 6/np.pi * np.arcsin(pearson_rank/2)
        
        if matrix:
            emp_norm_rank_corr = [[1,emp_norm_rank_corr],[emp_norm_rank_corr,1]]
        
        return emp_norm_rank_corr
    
    
    
    def emp_rank_corr(self, matrix = False):
        """Calculate the empirical rank correlation (Spearman's rho)
        
        If matrix, returns empirical rank correlation 
        in matrix form instead of single number. 
        
        
        Parameters
        ----------
        matrix : bool, optional
            return matrix instead of single number.
            
        """
        x = self.x
        y = self.y
        
        emp_rank_corr = sp.stats.spearmanr(x,y).correlation
        
        # Alternative method
        # rank_x = sp.stats.rankdata(x)/(len(x)+1)
        # rank_y = sp.stats.rankdata(y)/(len(y)+1)
        # emp_rank_corr = sp.stats.pearsonr(rank_x, rank_y)[0]
        
        if matrix:
            emp_rank_corr = [[1,emp_rank_corr],[emp_rank_corr,1]]
        
        return emp_rank_corr
    
    
    def emp_corr(self, matrix = False):
        """Calculate the empirical correlation (Pearson's r)
        
        If matrix, returns empirical correlation 
        in matrix form instead of single number. 
        
        
        Parameters
        ----------
        matrix : bool, optional
            return matrix instead of single number.
            
        """
        x = self.x
        y = self.y
        
        emp_corr = sp.stats.pearsonr(x,y)[0]
        
        if matrix:
            emp_corr = [[1,emp_corr],[emp_corr,1]]
        
        return emp_corr

    def fit(self):
        """Fit the copula with correlation and data 
        
        Note: this could also come directly from emp_corr
        
        
        """
        x = self.x
        y = self.y
        copula = self.copula
        fit_data = np.transpose(np.array([x,y]))
        copula.fit(fit_data)
        self.copula = copula
        self.fitted = True
        
    def check_fit(self):
        """Check whether the copula has been fitted yet. 
        """
        if not self.fitted:
            raise Exception("Copula not yet fitted. Fit with mycopula.fit().")
        
    # might work with different correlation
        
    """
    def fit2(self):
        #Fit the copula with correlation and data 
        
        Note: this could also come directly from emp_corr
        
        This apparently doesn't work
        
        
        copula = self.copula
        corr = self.emp_rank_corr(matrix = True)
        copula.set_corr(corr)
        self.copula = copula
    """
    
    def sample(self, n = 1000):
        """Generate sample values from the copula.
        
        Parameters
        ----------
        n : int
            The number of samples
        
        """
        self.check_fit()
        print("Sampling Gaussian copula.")
        #time0 = time.time()
        copula = self.copula
        self.sample_values = simulate(copula, n)
        #time1 = time.time()
        #print("Sampling took {} seconds".format(time1-time0))
        
        
    def unif2unifs(self, unival, samp_range = 0.005):
        """From the uniform value to the uniform values of the other parameter
        
        Note1: sample beforehand.
        Note2: it goes to unifS because of all the samples that are returned.
        Note3: the input axis should not matter because the symmetry of
            the copula. However, it does pose an inconsistency because of the
            samples that are taken from different sides. 
            
        Parameters
        ----------
        unival : float
            The uniform (0-1) value 
        samp_range : float (optional)
            The range from the value that is being sampled. Width = 2 times
            the samp_range. Default value 0.005.
            
        
        Returns
        -------
        univals : numpy.array of floats
            An array of the values (float) that are sampled and are within 
            the range of the samp_range 
            
        """
        
        try:
            sample_values = self.sample_values
            
        except ValueError:
            print("Sample values not assigned. Please first sample the copula.")
        
        univals = [y for x,y in sample_values 
            if x < unival+samp_range and x > unival-samp_range]
        
        return univals
            
        
    def val2unif(self, val, axis):
        """Calculate the uniform variant of a parameter value
        
        Parameters
        ----------
        val : float
            The value of the parameter
        axis : str
            The axis, either "x" or "y"
        
        Returns
        -------
        unival : float
            The uniform value
        """
        
        if axis == "x":
            ecdf = self.ecdf_x
        elif axis == "y":
            ecdf = self.ecdf_y
        else:
            print("Axis should either be 'x' or 'y'")
        
        unival = ecdf.__call__(val)
        return unival

    
    def unif2val(self, unival, ecdf):
       """Get the value from a single uniform value
       
       Parameters
       ----------
       unival : float
           Uniform value
       ecdf : ECDF() function
       
       Returns
       -------
       val : float
           Single uniform value
       """
       diff_search_val = np.abs(unival-ecdf.y)
       y_pos = np.nanargmin(diff_search_val)
       
       # ignore minus infinity
       if y_pos == 0:
           y_pos = 1
       val = ecdf.x[y_pos]
       #TODO: val correct?
       return val
   
    def unif2val_fit(self, unival, fitform):
       """Get the value from a single uniform value
       
       Parameters
       ----------
       unival : float
           Uniform value
       ecdf : ECDF() function
       
       Returns
       -------
       val : float
           Single uniform value
       """
       #TODO
       val = None
       return val
   
    
    def cond_expected_value_sample(self, val, cond_axis, samp_range=0.005,
                                   exp_meth='median'):
        """Calculate the conditional expected value
            (mean of the sampled values) from one 
            parameter to the other
        
        Parameters
        ----------
        val : float
            Floating value with the first 
        cond_axis : str
            The axis from where the condition comes from
        exp_meth : str
            Expected value method, should be either 'median' or 'mean'
            Default value: 'median'
        Returns
        -------
        exp_val : float
            Expected value
        
        """
        unival = self.val2unif(val, cond_axis)
        univals = self.unif2unifs(unival, samp_range=samp_range)
        if cond_axis == "x":
            new_axis = "y"
        elif cond_axis == "y":
            new_axis = "x"
        vals = self.unifs2vals(univals, axis=new_axis)

        if exp_meth.lower()=='mean':
            exp_val = np.nanmean(vals)
        elif exp_meth.lower()=='median':
            exp_val = np.nanmedian(vals)
        else:
            raise ValueError("exp_meth should either be 'mean' or 'median'")
        return exp_val
    
    
    def cond_std_sample(self, val, cond_axis, samp_range=0.005):
        """Calculate the conditional standard deviation from one 
            parameter to the other
        
        Parameters
        ----------
        val : float
            Floating value with the first 
        cond_axis : str
            The axis from where the condition comes from
        
        Returns
        -------
        std : float
            Standard deviation
        
        """
        unival = self.val2unif(val, cond_axis)
        univals = self.unif2unifs(unival, samp_range=samp_range)
        if cond_axis == "x":
            new_axis = "y"
        elif cond_axis == "y":
            new_axis = "x"
        vals = self.unifs2vals(univals, axis=new_axis)
        std = np.nanstd(vals)
        return std
    
    def cond_expected_value_pdf(self, val, cond_axis, n = 500):
        """ Calculate the conditional expected value
            (mean of the sampled values) from one 
            parameter to the other without using samples
        
        Parameters
        ----------
        val : float
            Floating value with the first 
        cond_axis : str
            The axis from where the condition comes from
        
        Returns
        -------
        exp_val : float
            Expected value
        
        #TODO: conditional std according to this method
        -> is that even possible. Probably yes, but quite cumbersome
        -> maybe not with this method...
        
        """
        self.check_fit()
        unival = self.val2unif(val, cond_axis)
        copula = self.copula
        sample_x = np.linspace(0,1,n)
        if cond_axis == "x":
            new_axis = "y"
        elif cond_axis == "y":
            new_axis = "x"
        pdfvals = np.array([copula.pdf(np.array([i,unival])) for i in sample_x])
        sc_pdfvals = pdfvals/np.nansum(pdfvals)
        sc_pdfvals = np.nan_to_num(sc_pdfvals)
        
        ecdfvals = np.array(self.unifs2vals(sample_x, new_axis))
        ecdfvals[ecdfvals == np.NINF] = 0
        ecdfvals[ecdfvals == np.inf] = 0
        ecdfvals[np.isnan(ecdfvals)] = 0
        
        exp_val = np.nansum(np.multiply(sc_pdfvals, ecdfvals))
        return exp_val
    
    
        
    def set_data_names(self, x_name=None, y_name=None):
        """Set the names of the data. Useful for documenting in 
            which what is saved
            
        Parameters
        ----------
        x_name : str
            Name of the x-data
        y_name : str
            Name of the y-data
        
        """
        if x_name is not None:
            self.x_name = x_name
        
        if y_name is not None:
            self.y_name = y_name
    
    def axis_from_data_name(self, name):
        """Get the build-in axis name (x or y) from the assigned name
        
        Parameters
        ----------
        name : str
            Name of the axis
        
        Returns
        -------
        axis : str
            Build-in name (x or y)
        """

        try:
            if self.x_name == name:
                axis = "x"
            elif self.y_name == name:
                axis = "y"
            else:
                print("Could not find an axis with this name.")
        except ValueError:
            print("You did not assign the axis a name")
        return axis
        
            
    def uncond_expected_value(self, axis):
        """Calculate the unconditional expected value
            (mean of the sampled values) of one dataset
        
        Parameters
        ----------
        axis : str
            Either "x" or "y"
        
        Returns
        -------
        exp_val : float
            Expected value (mean) of dataset
        
        """
        if axis == "x":
            exp_val = np.nanmean(self.x)
        elif axis == "y":
            exp_val = np.nanmean(self.y)
        return exp_val

    
    
    def unifs2vals(self, univals, axis = "y"):
        """Find the mean value from the uniform values
        
        Note: it goes from unifS because of all the samples that are put in
        
        Parameters
        ----------
        univals : numpy.array with floats
            Uniform values
        axis : str (optional)
            The axis, either "x" or "y"
        
        Returns
        -------
        vals : numpy.array of floats
            Values from uniform values
        """
        
        if axis == "y":
            ecdf = self.ecdf_y
        elif axis == "x":
            ecdf = self.ecdf_x
        else:
            print("Axis should either be 'x' or 'y'")
            
        """    
        vals = [self.unif2val(unival, ecdf) 
            if not np.isnan(self.unif2val(unival, ecdf)) else 0 for unival in univals]
        """
        
        vals = [self.unif2val(unival, ecdf) for unival in univals]
        
        return vals
    
    
    
    def uncond_std(self, axis):
        """Calculate the unconditional standard deviation
            of one dataset
        
        Parameters
        ----------
        axis : str
            Either "x" or "y"
        
        Returns
        -------
        std : float
            Standard deviation of dataset
        
        """
        if axis == "x":
            std = np.nanstd(self.x)
        elif axis == "y":
            std = np.nanstd(self.y)
        return std
    

class CopulaNd:
    """An empirical, two-dimensional, Gaussian copula with several useful methods.
    
    self.univals and self.vals save as follows: 
        self.vals.append([values, value_params, expect_params, vals])
    ...
    
    
    Attributes
    ----------
    data : numpy.array  (obligatory)
        input data emperical copula
        
    names : list of strings with len(dim) (optional)
        Manual names of the data
    
    dim : dimension 
        The number of parameters    
    
    copula : GaussianCopula() object
        A copula that is used 
    
    fitted : bool 
        Is the copula fitted yet?
        
    ecdfs : list of ECDF() objects
        ECDFs of all the parameters
        
    univals : list 
        List with saved conditional uniform values, shaped as:
            [values, value_params, expect_params, univals]
    
    vals : list
        List with saved conditional values, shaped as:
            [values, value_params, expect_params, vals]
    
    Methods
    -------
    axes_from_data_names(names)
        Get the build-in indices from the assigned names

    big_c(u)
        CDF of the copula
       
    check_fit()
        Check whether the copula has been fitted yet.
   
    cond_expected_value_pdf(values, expect_param, n=500)
        Conditional expected value of one parameter based on given values for all other parameters
   
    cond_sample(values, value_params, expect_params, n=500, save_univals=False, save_vals=False)
        Conditionally sample a copula with this method:  
           https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
           You can condition on not all the parameters. Conditioning is done by sampling, 
           samples can be saved separately.
       
        Make sure the parameters are ordered from small to large
   
    fit(method='cmle', verbose=True, ignore_message=False, **kwargs)
        Fit the Gaussian copula with specified data.
   
    plot_cond_samples(method='scatter', save_fig=False, save_path='conditional samples.png')
   
    sample(n)
        Generates random variables with selected copula's structure.
   
    set_corr(R)
        Set the correlation matrix of the copula
        Can also be done directly from Uninet (CorrMatrix.R)
        Can be unsaturated
       
    set_data_names(names)
        Set the names of the data

    small_c(u)
        PDF of the copula

    """


    def __init__(self, data):
        self.data = data
        self.dim = len(data[:,0])
        self.copula = GaussianCopula()
        self.fitted = False
        self.ecdfs = [ECDF(parameterdata) for parameterdata in data]
        self.univals = []
        self.vals = []
        
    def fit(self, method='cmle', verbose=True, ignore_message=False, **kwargs):
        """
		Fit the Gaussian copula with specified data.

		Parameters
		----------
		method : str (optional)
			The estimation method to use. Default is 'cmle'.
		verbose : bool (optional)
			Output various informations during fitting process.
        ignore_message : bool (optional)
            Whether ore not to ignore "Fitting the copula"
		**kwargs
			Arguments of method. See estimation for more details.

		Returns
		-------
		float
			The estimated parameters of the Gaussian copula.
		"""
        dim = self.dim
        X = np.transpose(self.data)
        if not ignore_message:
            print("Fitting Gaussian copula.")
        n = X.shape[0]
        if n < 1:
            raise ValueError("At least two values are needed to fit the copula.")
		#self._check_dimension(X[0, :])

		# Canonical Maximum Likelihood Estimation
        if method == 'cmle':
			# Pseudo-observations from real data X
            pobs = []
            for i in range(dim):
                order = X[:,i].argsort()
                ranks = order.argsort()
                u_i = [ (r + 1) / (n + 1) for r in ranks ]
                pobs.append(u_i)

            pobs = np.transpose(np.asarray(pobs))
			# The inverse CDF of the normal distribution (do not place it in loop, hungry process)
            ICDF = norm.ppf(pobs)

            def log_likelihood(rho):
                S = np.identity(dim)

				# We place rho values in the up and down triangular part of the covariance matrix
                rhocount=0
                rholen = len(rho)
                while rhocount < rholen:
                    for i in range(dim):
                        for j in range(dim):
                            if i != j:
                                if S[i][j] == 0.0:
                                    S[i][j] = rho[rhocount]
                                    S[j][i] = S[i][j]
                                    rhocount+=1
             
				# Computation of det and invert matrix
                if dim == 2:
                    RDet = S[0, 0] * S[1, 1] - rho**2
                    RInv = 1. / RDet * np.asarray([[ S[1, 1], -rho], [ -rho, S[0, 0] ]])
                else:
                    RDet = np.linalg.det(S)
                    RInv = np.linalg.inv(S)
				
				# Log-likelihood
                lh = 0
                for i in range(n):
                    cDens = RDet**(-0.5) * np.exp(-0.5 * np.dot(ICDF[i,  :], np.dot(RInv, ICDF[i,  :])))
                    lh += np.log(cDens)

                return -lh

            rho_start = [ 0.01 for i in range(int(dim * (dim - 1) / 2)) ]
            res = estimation.cmle(log_likelihood,
				theta_start=rho_start, theta_bounds=None,
				optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'),
				bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
            rho = res['x']
        elif method == 'mle':
            rho_start = [ 0.01 for i in range(int(dim * (dim - 1) / 2)) ]
            res, estimationData = estimation.mle(X, marginals=kwargs.get('marginals', None),
                                                 hyper_param=kwargs.get('hyper_param', None),
                                                 hyper_param_start=kwargs.get('hyper_param_start', None), hyper_param_bounds=kwargs.get('hyper_param_bounds', None), theta_start=rho_start, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
            rho = res['x']
			
        R = np.identity(dim)
		# We extract rho values to covariance matrix
        rhocount=0
        rholen = len(rho)
        while rhocount < rholen:
            for i in range(dim):
                for j in range(dim):
                        if i != j:
                               if R[i][j] == 0.0:
                                   R[i][j] = rho[rhocount]
                                   R[j][i] = R[i][j]
                                   rhocount+=1

        R = math_misc.nearPD(R)
        self.set_corr(R)

        #self.copula.set_corr(R)
        #return R
    

        
    def check_fit(self):
        """Check whether the copula has been fitted yet. 
        """
        if not self.fitted:
            raise Exception("Copula not yet fitted. Fit with mycopula.fit().")
    
    
    def set_corr(self, R):
        """ Set the correlation matrix of the copula
        Can also be done directly from Uninet (CorrMatrix.R)
        Can be unsaturated
        
        Parameters
        ----------
        R : numpy.Ndarray
            The N x N correlation matrix
        """
        
        self.R = R
        self.fitted=True        
        
    def set_data_names(self, names):
        """Set the names of the data
        
        Parameters
        ----------
        names : list or array of strings
            The names of the parameters. Needs to have
            the length of the dimension.
        
        
        """
        
        if len(names) != self.dim:
            print("[!] Could not set the names. The length of the array with names is not equal to the dimension of the copula.")
            
        else:
            self.names = names
            
    def axes_from_data_names(self, names):
        """Get the build-in indices from the assigned names
        
        Parameters
        ----------
        names : list of str
            Names of the axis
        
        Returns
        -------
        axes : list
            Indices
        """
        axes = []
        for name in names:
            if name in self.names:
                axes.append(self.names.index(name))
            else:
                print(f"[!] Could not find \"{name}\" in CopulaNd.names. Index not given")
        return axes
    
    def small_c(self, u):

        """PDF of the copula
        
        Parameters
        ----------
        u : numpy.array of floats with dimension 1 * dim
            Parameters of all the
        
        Returns
        -------
        Value of pdf at given point
            
        """  
        matR = self.R
        X = norm.ppf(u)
        a = np.linalg.det(matR)**(-0.5)
        b = np.exp(-0.5*np.dot(np.transpose(X),np.dot(np.linalg.inv(matR)-np.eye(len(u)), X)))
        return a*b        
            
    def big_c(self, u):
        """CDF of the copula
        
        Parameters
        ----------
        u : numpy.array of floats with dimension 1 * dim
            Parameters of all the
        
        Returns
        -------
        Value of CDF at given point
        
        
        \Phi _{R} is the joint cumulative distribution function of a multivariate normal 
        distribution with mean vector zero and covariance matrix equal to the correlation matrix
        
        """
        matR = self.R
        val = mvn.cdf(norm.ppf(u), cov=matR)
        return val
    
    def sample(self, n):
        """
        Generates random variables with selected copula's structure.
    
        Parameters
        ----------
    	n : integer
    		The size of the sample.
            
        Returns
        -------
        X : list (n*dim)
            The samples.
    	"""
        self.check_fit()
        matR = self.R
        d = self.dim
    	
        X = []
        # We get correlation matrix from covariance matrix
        A = cholesky(matR)

        for i in range(n):
            Z = np.random.normal(size=d)
            V = np.dot(A, Z)
            U = norm.cdf(V)
            X.append(U)
        
        return X
    
    def cond_expected_value_pdf(self, values, expect_param, n=500):
        """Conditional expected value of one parameter based on given values for all other parameters
        
        Parameters
        ----------
        values : numpy.array (len = dim-1)
            Values of all other parameters 
        
        exp_param : int
            The index (0 indexed) of the conditional parameter
            
        n : int (optional)
            The number of regular  samples to use. Default value: 500
            
        Returns
        -------
        exp_val : float
            The conditional expected value
        """
        self.check_fit()
        dim = self.dim
        ecdfs = self.ecdfs
        
        values = np.insert(values, expect_param, 0)
          
        univals = [[ecdfs[i].__call__(values[i]) for i in range(dim)]]

        factors = np.zeros(n)
        for i in range(n):
            u3 = i/n
            univals[0][expect_param] = u3
            factors[i] = self.small_c(np.transpose(univals))
        
        vals_pre = np.array([Copula2d.unif2val(self, unival, ecdfs[expect_param]) for unival in np.linspace(0,1,n)])
        
        #remove NaNs etc:
        vals_pre[vals_pre == np.NINF] = np.nan
        vals_pre[vals_pre == np.inf] = np.nan
        
        fact2 = factors/np.nansum(factors)
               
        exp_val = np.nansum(np.multiply(fact2, vals_pre))
        return exp_val
    
    def _mu_hat(self, a,Ronetwo, Rtwotwo):
        """ Caluclate mu hat from 
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions           
        for cond_sample
        """
        mu_hat = np.dot(np.dot(Ronetwo, np.linalg.inv(Rtwotwo)),a) 
        
        return mu_hat
    
    def _R_hat(self, Roneone, Ronetwo, Rtwoone, Rtwotwo):
        """ Caluclate R hat from 
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions           
        for cond_sample
        """
        
        R_hat = Roneone - np.dot(Ronetwo, np.dot(np.linalg.inv(Rtwotwo), Rtwoone))
        return R_hat 
    

class MultVarNorm:
    """A multivariate normal item.
    
    Multivariate normal calculation item of a Bayesian Network.
    
    ...
    
    
    Attributes
    ----------
    data : numpy.array  (obligatory)
        input data emperical copula
        
    names : list of strings with len(dim) (optional)
        Manual names of the data
    
    dim : dimension 
        The number of parameters    
    

    Methods
    -------
    axes_from_data_names(names)
        Get the build-in indices from the assigned names

    """


    def __init__(self, data):
        self.data = data
        self.dim = len(data[:,0])
        self.fitted = False
        self.ecdfs = [ECDF(parameterdata) for parameterdata in data]
        self.vals = []
    
    
    def set_corr(self, R):
        """ Set the correlation matrix of the copula
        Can also be done directly from Uninet (CorrMatrix.R)
        Can be unsaturated
        
        Parameters
        ----------
        R : numpy.Ndarray
            The N x N correlation matrix
        """
        
        self.R = R
        self.fitted=True        
        
    def set_data_names(self, names):
        """Set the names of the data
        
        Parameters
        ----------
        names : list or array of strings
            The names of the parameters. Needs to have
            the length of the dimension.
        
        
        """
        
        if len(names) != self.dim:
            print("[!] Could not set the names. The length of the array with names is not equal to the dimension of the copula.")
            
        else:
            self.names = names
            
    def axes_from_data_names(self, names):
        """Get the build-in indices from the assigned names
        
        Parameters
        ----------
        names : list of str
            Names of the axis
        
        Returns
        -------
        axes : list
            Indices
        """
        axes = []
        for name in names:
            if name in self.names:
                axes.append(self.names.index(name))
            else:
                print(f"[!] Could not find \"{name}\" in CopulaNd.names. Index not given")
        return axes
    
    def _mu_hat(self, a,Ronetwo, Rtwotwo):
        """ Caluclate mu hat from 
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions           
        for cond_sample
        """
        mu_hat = np.dot(np.dot(Ronetwo, np.linalg.inv(Rtwotwo)),a) 
        
        return mu_hat
    
    def _R_hat(self, Roneone, Ronetwo, Rtwoone, Rtwotwo):
        """ Caluclate R hat from 
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions           
        for cond_sample
        """
        
        R_hat = Roneone - np.dot(Ronetwo, np.dot(np.linalg.inv(Rtwotwo), Rtwoone))
        return R_hat 
    
    
    def sig_ad(self, x, *a):
        """Run Sigmoid function.
        
        Defined as ...

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        *a : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        poly = [a[i+3]*(x-a[2])**(2*i+1) for i in range(len(a)-3)]
        P = np.sum(poly, 0)
        return 1/(1+np.exp((P-a[0])*a[1])) 
    
    def stretchify(self, x,f, mid):
        """Stretch the distribution at the end.
        

        Parameters
        ----------
        x : float-like or array-like
            Value between 0 or 1.
        f : float
            factor used. Most useful is between 2 and pi.

        Returns
        -------
        similar to x
            Changed values.

        """
        
        if f == 0:
            return x
        else:
            return (np.tan((x-0.5)*f)/np.tan(0.5*f)*0.5+0.5)*((x-mid)**2) + x * (1-(x-mid)**2) 
        
    def mult_gauss_mixt_cdf(self, X, *params, f = 0, mid = 0.5):
        vals = []
        #print(params)
        if len(params)%3 != 0:
            raise ValueError("[!] params should be divisable by 3")
        params = np.reshape(params, (-1,3))
        for i in range(len(params)):
            par = params[i]
            vals.append(par[2]*norm.cdf(X, loc = par[0], scale = par[1]))
        valssum = np.sum(vals, 0)/np.sum(params, 0)[2]
        valssum = self.stretchify(valssum, f, mid = mid)
        return valssum
    
    def _mult_gauss_mixt_pdf_plot(self, X, *params):
        vals = []
        if len(params)%3 != 0:
            raise ValueError("[!] params should be divisable by 3")
        params = np.reshape(params, (-1,3))
        for i in range(len(params)):
            par = params[i]
            vals.append(par[2]*norm.pdf(X, loc = par[0], scale = par[1]))
        for valsunit in vals:
            plt.plot(X, valsunit)
    
    def fit_sigmoid(self, method = 'trf',
                  parlen = 6,
                  extra_up = 0.2, 
                  extra_down = 0.2,
                  numvals = 100000,
                  maxfev = 100000):
        """Fit the ECDFS of the sigmoid method.
        
        Fit a sigmoid function to the data
        
        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'trf'.
        p0 : TYPE, optional
            DESCRIPTION. The default is (-1,1.00, -1., -1., -0.1, -0.1).
        bounds : TYPE, optional
            DESCRIPTION. The default is ((-np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf),(np.inf, np.inf, np.inf, 0, 0, 0)).
        numvals : int, optional
            Numver of values for the interpolate formula. Default value is 100000.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.fit_params = []
        self.interpols = []
        
        
        
        for ecdf in self.ecdfs:
            x = ecdf.x[1:]
            rangeval = ecdf.x[-1]-ecdf.x[1]
            minval = ecdf.x[1]-extra_down*rangeval 
            maxval = ecdf.x[-1]+extra_up*rangeval
            

            #TODO do something with limited
            p0 = ((minval+rangeval/2)*0.2,0.07/rangeval, minval+rangeval/2, -80/rangeval**0.5, -1/rangeval**0.5, -0.01/rangeval**2)
            bounds = ((-np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf),(np.inf, np.inf, np.inf, 0, 0, 0))
            if parlen > 6:
                raise ValueError("[!] parlen could be 6 at max")
            else:
                if parlen < 6:
                    p0 = tuple(list(p0)[:parlen])
                    boundslist = list(bounds)
                    boundsnew = []
                    for i in boundslist:
                        boundsnew.append(tuple(list(i)[:parlen]))
                    bounds = tuple(boundsnew)

            try:
                popt, pcov = curve_fit(self.sig_ad, x, ecdf.y[1:], p0=p0,
                                method = method,
                                bounds = bounds
                                )
                self.fit_params.append(popt)
            except:
                popt = None
                self.fit_params.append(None)
                print("It didn't work for this parameter")
            
            # plt.figure()
            # plt.plot(ecdf.x, ecdf.y)
            # plt.plot(x, self.sig_ad(x, *p0))
        
            
            interpol = self._inter_funcs(self.sig_ad, *tuple(popt),
                                         numvals = numvals, minval = minval, maxval = maxval)
            # plt.figure()
            # plt.plot(ecdf.x, ecdf.y)
            # plt.plot(ecdf.x, self.sig_ad(ecdf.x, *tuple(popt)))
            self.interpols.append(interpol)
            
    def fit_mix_gauss(self, num_gauss = 3, numvals = 100000,
                      extra_up = 0.2, f = 0,
                      mid = 0.5,
                      extra_down = 0.2,
                      maxfev = 10000, verbose = 0):
        """ Fit mixed gaussian
        

        Parameters
        ----------
        num_gauss : TYPE, optional
            DESCRIPTION. The default is 3.
        numvals : TYPE, optional
            DESCRIPTION. The default is 100000.
        extra_up : TYPE, optional
            DESCRIPTION. The default is 0.2.
        extra_down : TYPE, optional
            DESCRIPTION. The default is 0.2.
        maxfev : TYPE, optional
            DESCRIPTION. The default is 10000.

        Returns
        -------
        None.

        """
        # TODO: do something with n and aic/bic 
        self.fit_params = []   
        self.interpols = []
        p0s = []
        for i in range(self.dim):
            ecdf = self.ecdfs[i]
            
            rangeval = ecdf.x[-1]-ecdf.x[1]
            minval = ecdf.x[1]-extra_down*rangeval 
            maxval = ecdf.x[-1]+extra_up*rangeval
            p0 = tuple([(k+1.1)/1.008/num_gauss*rangeval+minval if j%3 == 0
                        else (k+3)/num_gauss*rangeval*0.15
                        if j%3 == 1
                            else (k+5)/6
                        for k in range(num_gauss) for j in range(3)
                        ])
            #print(p0)
            p0s.append(p0)
            
#            plt.figure()
#            plt.hist(ecdf.x[1:], density=True)
#            self._mult_gauss_mixt_pdf_plot(ecdf.x[1:], *p0)
            # bounds = (tuple([
            #         minval if j%3 == 0
            #         else 0.05 for i in range(num_gauss) for j in range(3)]), #TODO Check if changed worked!
            #         tuple([maxval if j%3 == 0
            #         else np.inf for i in range(num_gauss) for j in range(3)])
            #         )
            #TODO did it work
            bounds = (tuple([
                    minval if j%3 == 0
                    else rangeval/100 
                    if j%3 == 1 
                    else 0 for i in range(num_gauss) for j in range(3)]),
                    tuple([maxval if j%3 == 0
                    else np.inf for i in range(num_gauss) for j in range(3)])
                    )
            best_params, _ = curve_fit(self.mult_gauss_mixt_cdf, ecdf.x[1:], 
                                       #(ecdf.y[1:]-0.5)*0.95+0.5, p0 = p0, #!!! hier heb ik wat veranderd!
                                       ecdf.y[1:], p0 = p0,
                                       maxfev = maxfev,
                                       bounds=bounds, verbose=verbose)
#            plt.figure()
#            plt.plot(ecdf.x, ecdf.y)
#            plt.plot(ecdf.x, self.mult_gauss_mixt_cdf(ecdf.x, *tuple(best_params)))
            #print(best_params)
            self.fit_params.append(best_params)
            
            # if i != 1:
            #     g = 0
            # else:
            #     g = f
            #if 
            if isinstance(mid, list) or isinstance(mid, np.ndarray):
                md = mid[i]
            else:
                md = mid
            interpol = self._inter_funcs(self.mult_gauss_mixt_cdf, *tuple(best_params),f=f,mid = md,
                                         numvals = numvals, minval = minval, maxval = maxval)
            self.interpols.append(interpol)
            #print(f"Parameter {i} succeeded!")
        return p0s    

    def _inter_funcs(self, function, *other_params, f = 0, numvals = 1000000, minval = 0, maxval = 100, mid = 0.5):
        xvals = np.linspace(minval, maxval, numvals)
        yvals = function(xvals, *other_params, f = f, mid = mid)
        return interpolate.interp1d(yvals, xvals, fill_value="extrapolate")    

    def cond_sample(self, values, value_params, expect_params, fit_func = 'sigmoid',
                        n=500, 
                    conf_int=0.682,
                    save_univals=False,
                    save_vals=False,
                    exp_meth='median'
                    ):
        """Conditionally sample a copula with MVNormal method.
            
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        You can condition on not all the parameters. Conditioning is done by sampling, 
        samples can be saved separately.
        
        Make sure the parameters are ordered from small to large

        
        Parameters
        ----------
        values : numpy.array or list of floats
            The conditioning values 
            
        value_params : numpy.array or list of ints
            The indices of the parameters of the values. Should have same length
            as the values.
            
        expect_params : numpy.array or list of ints
            The indices of the parameters you want to know the expected value of
            
        n : int (optional)
            The number of samples to calculate
            
        conf_int : float (optional)
            The confidence interval (always in the middle).
            The part of samples in the middle that lie within the given
            confidence interval.
            Default value: 0.682 (sigma = 1)
            
        save_univals : bool (optional)
            When True, save the uniform sample values to the class, as follows:
            values, value_params, expect_params, rand_mvn_un
            Default value: False
        
        save_vals : bool (optional)
            When True, save the calculated parameter sample values to the class, as follows:
            values, value_params, expect_params, vals
            Default value: False
        
        exp_meth : str
            Expected value method, should be either 'median' or 'mean'
            Default value: 'median'
            
        Returns
        -------
        expec : numpy.array
            For each expect_param, the expected value
            
        stdd : numpy.array
            For each expect_param, the standard deviation
            
        confidence_interval : numpy.array
            For each expect_param, the confidence interval:
                lower limit, upper limit
        
        """
        if len(values) != len(value_params):
            raise ValueError("values should have the same length as value_params")
            
        if len(values) + len(expect_params) > self.dim:
            raise ValueError("More parameters passed than dimension of BN")
            
        for expect_param in expect_params:
            if expect_param in value_params:
                raise ValueError("Expected parameter also given as input parameter")
            
        dim = self.dim
        R = self.R
        fp = self.fit_params
        
        if fit_func.lower() in ['sigmoid', 'logistic']:
            a = norm.ppf(np.transpose(np.array([[self.sig_ad(values[i], *tuple(fp[value_params[i]]))
                                                 for i in range(len(value_params))]])))
        elif fit_func.lower() in ['mixed gauss', 'mixgauss', 'mg', 'mixed gaussian']:
            a = norm.ppf(np.transpose(np.array([[
                            self.mult_gauss_mixt_cdf(values[i],
                            *tuple(fp[value_params[i]]))
                            for i in range(len(value_params))]])))
        elif fit_func.lower() == 'ecdf':
            a = norm.ppf(np.transpose(np.array([[self.ecdfs[value_params[i]].__call__(values[i])
                                                 for i in range(len(value_params))]])))
        
        else:
            raise ValueError("fit_func should be either 'sigmoid', 'mixgauss' or 'ecdf'")
                       
            
        a[a>3.5] = 3.5  # 0.9997673709209645 = 1 - 1/4300
        a[a<-3.5] = -3.5  # 0.9997673709209645 = 1- 1/4300

        others = [i for i in range(dim) if i not in value_params]

        # Create subsets of R
        Roneone = R[tuple(np.meshgrid(others,others))].copy()        
        Rtwoone = R[tuple(np.meshgrid(value_params,others))].T.copy()
        Ronetwo = R[tuple(np.meshgrid(others,value_params))].T.copy()
        Rtwotwo = R[tuple(np.meshgrid(value_params,value_params))].copy()
        
        
        mu_hat = self._mu_hat(a, Ronetwo, Rtwotwo)
        R_hat = self._R_hat(Roneone, Ronetwo, Rtwoone, Rtwotwo)
        
        rand_mvn = np.random.multivariate_normal(mu_hat.T[0], R_hat, size=n)       
        rand_mvn_un = norm.cdf(rand_mvn)
        
        if save_univals:
            self.univals.append([values, value_params, expect_params, rand_mvn_un.T])
        
        
        if fit_func.lower() in ['sigmoid', 'logistic'] or fit_func.lower() in ['mixed gauss', 'mixgauss', 'mg', 'mixed gaussian']:
            vals = [self.interpols[i](rand_mvn_un[:,others.index(i)])
                    for i in expect_params]
        elif fit_func.lower() == 'ecdf':
            vals = [[Copula2d.unif2val(self, unival, self.ecdfs[i]) 
                        for unival in rand_mvn_un[:,others.index(i)]] 
                        for i in expect_params]

        vals = np.array(vals)
        vals[vals==np.inf] = np.nan
        vals[vals==-np.inf] = np.nan
        
        if save_vals:
            self.vals.append([values, value_params, expect_params, vals])
        
        if exp_meth.lower()=='mean':
            expec = np.nanmean(vals, axis=1)
        elif exp_meth.lower()=='median':
            expec = np.nanmedian(vals, axis=1)
        else:
            raise ValueError("exp_meth should either be 'mean' or 'median'")
        standd = np.nanstd(vals, axis=1)
        
        #confidence margin
        confmargin = (1-conf_int)/2
        confidence_interval = [np.quantile(i, [confmargin, 1-confmargin]) for i in vals]
        
        return expec, standd, confidence_interval
         

    
class Predict:
    """Make a prediction from the data and test it.
    
    ...
    
    
    Attributes
    ----------
    data : numpy.array
        The input data (conditional and target values) with shape (dim*len)
    
    pred_axes : int
        Which axes to predict
        
    train_data : numpy.array
        Training set. The same as `data`, except when train_test_set is run.
        
    test_data : numpy.array
        Training set. The same as `data`, except when train_test_set is run.
    
    
    Methods
    -------
    single_cop(pred_meth = 'pdf', n='1000', samp_range = 0.005)
        Make prediction for single copula.
        
    network()
        Make prediction for Bayesian network
        
    nse()
        Nash-Sutcliffe Efficiency. 
        
    kge()
        Kling-Gupta Efficiency.
        
    
    """
    
    def __init__(self, data, pred_axes, R = None):
        self.data = data.copy()
        self.train_data = data.copy()
        self.test_data = data.copy()
        self.pred_axes = pred_axes
        if R is not None:
            self.R = R
        
    #TODO
    def test_multiple_times(self,
                            cop_or_net,
                            n_tests,
                            test_size=None,
                            train_size=None,
                            pred_meth = 'pdf',
                            n='1000',
                            samp_range = 0.005
                            ):
        if cop_or_net.lower() in ['single', 'copula', 'cop']:
            params = np.zeros(n_tests)
            for i in range(n_tests):
                params[i] = i
        
        return params
            
        
    def train_test_set(self, 
                   test_size=None, 
                   train_size=None,
                   random_state=None,
                   ):
        
        inp_data = np.transpose(self.data)
        train_data_pre, test_data_pre = train_test_split(inp_data, 
                                                         test_size=test_size,
                                                         train_size=train_size,
                                                         shuffle=True)
        train_data = np.transpose(train_data_pre)
        test_data = np.transpose(test_data_pre)
        self.train_data = train_data
        self.test_data = test_data
        return train_data, test_data

    def single_cop(self,
               pred_meth = 'pdf',
               n=1000,
               samp_range = 0.005):
        
        """Make a prediction from a single copula
        
        Parameters 
        ----------
        pred_meth : str (optional)
            Default value: 'pdf'. Prediction method. For 2d-case, can also
            be 'sample'.
             
        n : int (optional)
            The number of samples. Either the number of random samples 
            (pred_meth = 'sample') or the number of regular samples 
            (pred_meth = 'pdf')
            Default value = 1000
        
        samp_range : float (optional)
            When 2 variables and pred_meth = 'sample', 0.5 * the width of 
            samples to take into account.
        
        Returns
        -------
        pred_coef : float
            The coefficient of fit
        
        """
        train_data = self.train_data
        test_data = self.test_data
        if isinstance(self.pred_axes, int):
            pred_axis=self.pred_axes
        elif len(self.pred_axes) == 1:
            pred_axis = self.pred_axes[0]
        else:
            raise ValueError("For single_cop, self.pred_axes should either be an integer or a list/array of length 1")
        
        
        # Handle data
        lendata = len(test_data[0])
        target = test_data[pred_axis]
        
        # Make prediction
        if np.min(train_data.shape) == 2:
            # take the two-dimensional one
            
            # train
            copula = Copula2d(train_data[int(np.abs(pred_axis-1))], train_data[pred_axis]) # always predict the y-axis
            copula.fit()
                       
            # predict
            if pred_meth.lower() == 'pdf':
                prediction = np.array([copula.cond_expected_value_pdf(test_data[int(np.abs(pred_axis-1))], 'x', n=n) for i in range(lendata)])
                
            elif pred_meth.lower() == 'sample':
                copula.sample(n=n)
                # predict
                prediction = np.array([copula.cond_expected_value_sample(test_data[int(np.abs(pred_axis-1))], 'x', samp_range=samp_range) for i in range(lendata)])
        
        elif np.min(train_data.shape) > 2:
            # take the n-dimensional one
            copula = CopulaNd(train_data)
            copula.fit()
            input_vals = np.transpose(np.delete(test_data, pred_axis, axis=0))
            prediction = [copula.cond_expected_value_pdf(input_vals[i], pred_axis,n=n) for i in range(lendata)]

        else:
            print("[!] Data should have multiple dimensions.")
              
        self.target = np.array([target])
        self.prediction = np.array([prediction])
        return prediction
        
    def bn(self, fit_func = 'sigmoid',
            value_params=None, 
            n = 1000, 
            f = 0,
            mid = 0.5,
            conf_int = 0.682,
            exp_meth = 'median', method = 'trf',
            numpars = 6,
            extra_up = 0.1, extra_down=0.1,
            maxfev = 100000,
            numvals = 1000000):
        
        """Calculate multiple predictions with input data from a saturated or unsaturated copula. 
        
        Works with CopulaNd.cond_sample:
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        You can condition on not all the parameters. Conditioning is done by sampling.
        
        Parameters
        ----------
        R : numpy.array or numpy.matrix (optional)
            Correlation matrix of the network
        
        value_params : numpy.array, list or None (optional)
            From which axes to take the conditioning values
            When None, takes all axes not in self.pred_axes
            
        n : int (optional)
            Number of samples 
            Default value = 1000
        
        conf_int : : float (optional)
            The confidence interval (always in the middle).
            The part of samples in the middle that lie within the given
            confidence interval.
            Default value: 0.682 (sigma = 1)
            
        exp_meth : str
            Expected value method, should be either 'median' or 'mean'
            Default value: 'median'
            
        Returns
        -------
        exp_array : numpy.array
            Expected values array
            timestep:parameter
            
        conf_array : numpy.array
            Confidence interval array
            timestep:parameter:lower,upper
        """
        try:
            R = self.R
        except:
            raise ValueError("Predict method should have an self.R in order to use this method.")
        time0 = time.time()
        train_data = self.train_data
        test_data = self.test_data
        pred_axes = self.pred_axes
        
        if value_params is None:
            value_params = [i for i in range(len(train_data)) if i not in pred_axes]
        
        #print("Training this MVN cannot be done yet")
        
        
        mvn = MultVarNorm(train_data)
        mvn.set_corr(R)
        
        if fit_func.lower() in ['sigmoid', 'logistic']:
            mvn.fit_sigmoid(parlen = numpars, numvals = numvals, 
                                 extra_up = extra_up, extra_down=extra_down, 
                                 maxfev = maxfev)
        elif fit_func.lower() in ['mixed gauss', 'mixgauss', 'mg', 'mixed gaussian']:
            mvn.fit_mix_gauss(num_gauss = numpars, numvals = numvals, 
                             extra_up = extra_up, extra_down=extra_down, 
                             maxfev = maxfev, mid = mid,
                             f = f)
        elif fit_func.lower() == 'ecdf':
            mvn.fit_params = None
        
        else:
            raise ValueError("fit_func should be either 'sigmoid', 'mixgauss' or 'ecdf'")
        
        exp_array = np.zeros((len(test_data[0]),len(pred_axes)))
        conf_array = np.zeros((len(test_data[0]),len(pred_axes),2))
        
        for i in range(len(test_data[0])): 
            datastep = test_data[:,i][value_params]
            
            exp_vals, _, confidence_interval = mvn.cond_sample(
                                datastep, value_params,
                                pred_axes, n=n,
                                fit_func = fit_func,
                                conf_int = conf_int,
                                save_univals=False, 
                                save_vals=False,
                                exp_meth=exp_meth)
            
            exp_array[i] = exp_vals
            conf_array[i] = confidence_interval          
            
        self.target = test_data[pred_axes]
        self.prediction = exp_array.T
        self.conf_int = conf_array
        print(f"Calculating took {time.time()-time0} seconds")
        return exp_array, conf_array
            
    def bn_ecdf_input_confidence():
        """Similar to bn_ecdf but then with already an confidence interval
        of the measurement as an input
        
        Assumed to be independent
        
        Use error sum of squares somehow?
        
        Idea: 
            1. Calculate with confidence as input
            2. Calculate normal, confidence output
            3. Calculate error for both from expected value
            4. Error = sqrt(error1^2+error2^2)
        """
        # https://physics.stackexchange.com/questions/23441/how-to-combine-measurement-error-with-statistic-error?noredirect=1&lq=1
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
        pass
    
    
    def nse(self):
        """Nash-Sutcliffe Efficiency. Prediction data should be in class.
        """
        nse_ = np.zeros(len(self.prediction))
        for i in range(len(self.prediction)):
            try: 
                nse_item = he.nse(self.prediction[i], self.target[i])
                nse_[i] = nse_item
            except NameError:
                print("Prediction or target not defined")
        if len(nse_)==1:
            nse_ = nse_[0]
        return nse_
        
    def kge(self):
        """Kling-Gupta Efficiency. Prediction data should be in class. 
        
        Returns
        -------
        kge_val : numpy.array
            The coefficient.
        
        r : numpy.array
            r factor KGE
        
        alpha : numpy.array
            Alpha factor KGE
        
        beta : numpy.array
            Beta factor KGE
        """
        kge_val = np.zeros((len(self.prediction),4))
        for i in range(len(self.prediction)):
            try: 
                kge_array = he.kge(self.prediction[i], self.target[i]).T
                kge_val[i] = kge_array
            except NameError:
                print("Prediction or target not defined")
                
        if len(kge_val)==1:
            kge_val = kge_val[0]
        return kge_val
    
     
        
        
    def k_fold_ecdf(self, k = 5, n=5000, exp_meth='mean', conf_int = 0.682,
                         value_params = None):
        """
        """
        section = int(len(self.data[0])/k)
        indices = [i for i in range(k) for j in range(section)]
        c = 1
        while len(indices) < len(self.data[0]):
            indices.append(k-c)
            c+=1       
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        nses = []
        kges = []
        
        for i in range(k):
            boolind = indices == i
            self.train_data = self.data[:, ~boolind]
            self.test_data = self.data[:, boolind]
            self.bn_ecdf(self.R, value_params = value_params, n=n, 
                                   conf_int = conf_int,
                                   exp_meth = exp_meth)
            nses.append(self.nse())
            kges.append(self.kge())
        
        return nses, kges
    
        
    def k_fold_bn(self, k=5, fit_func = 'sigmoid', n=5000, exp_meth='mean', conf_int = 0.682,
                         value_params = None,
                         f = 0,mid = 0.5,
                         numvals = 1000000, 
                         extra_up = 0.1, extra_down=0.1, 
                         maxfev = 100000,
                         return_predictions = False,
                         numpars = 6):
        """K-fold cross validation for multivariate normal with mixed Gaussian fit.
        

        Parameters
        ----------
        k : int
            Number of folds.
        n : int, optional
            Number of samples to use for the monte carlo of the conditional
            multivariate normal. The default is 5000.
        exp_meth : str, optional
            Expected value method. Either 'mean' or 'median'.
            The default is 'mean'.
        conf_int : float, optional
            Width of confidence interval. All data = 1. The default is 0.682.
        value_params : List of ints, optional
            Which values to use for prediction. When None,
            imply all other values are used. The default is None.
        numvals : int, optional
            Number of values for interpolate method. The default is 1000000.
        extra_up : float, optional
            Part of the distribution to predict extra upwards.
            Data span = 1. The default is 0.1.
        extra_down : float, optional
            Part of the distribution to predict extra downwards. 
            Data span = 1. The default is 0.1.
        maxfev : int, optional
            Maximum iterations for curve fit. The default is 100000.
        num_gauss : int, optional
            Number of Gaussians for fit. The default is 4.

        Returns
        -------
        nses : list
            All the NSEs that have been found.
        kges : list
            All the KGEs that have been found.

        """
        section = int(len(self.data[0])/k)
        indices = [i for i in range(k) for j in range(section)]
        c = 1
        while len(indices) < len(self.data[0]):
            indices.append(k-c)
            c+=1       
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        nses = []
        kges = []
        predictions = []
        data = []
        for i in range(k): # try...
            boolind = indices == i
            self.train_data = self.data[:, ~boolind]
            self.test_data = self.data[:, boolind]
            self.bn(fit_func = fit_func, numpars = numpars, 
                    f = f, mid = mid,
                    value_params = value_params, n=n, 
                                   conf_int = conf_int,
                                   exp_meth = exp_meth,
                                   numvals = numvals, 
                                 extra_up = extra_up, extra_down=extra_down, 
                                 maxfev = maxfev)
            nses.append(self.nse())
            kges.append(self.kge())
            predictions.append(self.prediction[0])
            data.append(self.test_data[self.pred_axes[0]])
        if return_predictions:
            return nses, kges, predictions, data
        else:
            return nses, kges