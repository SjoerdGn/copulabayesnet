# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:08:34 2019

Taking montly maxima, averages, selecting data prior to 

@author: GNOS
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import easygui
import pandas as pd

#from tools.cop_plot import water_balance as wb_plot

import json
from calendar import monthrange
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "13"
        
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'



class DataProc:
    
    """
    Further process the data
    
    Give it a data_dict from a DataPreProc() object
    
    ...
    
    
    Attributes
    ----------
    data_dict : dictionary with pandas.DataFrames with the data
        Formatted as:
        name data : dataframe
    
    Methods
    -------
    load_processed_file(path=None, **kwargs)
        Load data that already has been 
        preproecessed and processed, to a DataFrame.
  
    merge(name_list=None, timestep='month', type_data='mean', combination='inner', export_name='merged')
        Merge dataframes
  
    resample_adv()
        Resamples with back_factors
  
    resample_basic(name_list=None, timestep='M', resamp_meth='mean', inplace=False, new_names=None)
        Resample basic: get one number per given timestep
  
    spatially_combine_data(new_name, lambdas, column=None, name_list=None)
        Combine multiple spatial data sources with factors
      
        Alphabetically order the weights!
  
    to_array(name=None)
        Return a certain DataFrame as an array
  
    to_csv(name_list=None, index=False)
        Save one or multiple dataframes to csv
        with a gui (when no names are passed)
      

    """
    
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def resample_basic(self, name_list=None, timestep = 'M', 
              resamp_meth = 'mean', inplace = False, new_names=None):
        
        """Resample basic: get one number per given timestep
        
        Parameters
        ----------
        name_list : list of strings or None (optional)
            Select the dataframes to combine. When None, select with GUI.
            
        timestep : str (optional)
            Default value: 'M' for month. Can also be any other timestep.
            
        resamp_meth: str (optional)
            The resample method. Default value: 'mean'. 
            Can also be 'max', 'min' or 'sum'.
            
        inplace : bool (optional)
            Wheter to replace the old dataset
            
        
        """
        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick the to resample dataframes",
                                              choices=[item for item in self.data_dict])
        count=0
        for name in name_list:
            if resamp_meth.lower() == 'mean':
                new_df = self.data_dict[name].resample(timestep).apply(np.nanmean)
            elif resamp_meth.lower() == 'max':
                new_df = self.data_dict[name].resample(timestep).apply(np.nanmax)
            elif resamp_meth.lower() == 'min':
                new_df = self.data_dict[name].resample(timestep).apply(np.nanmin)
            elif resamp_meth.lower() == 'sum':
                new_df = self.data_dict[name].resample(timestep).apply(np.sum)
            else:
                print("[!] resamp_meth should be either 'mean', 'max', 'min' or 'sum'.")
                   
            if inplace:
                self.data_dict[name] = new_df
            else:
                if new_names is None:
                    self.data_dict[name+timestep] = new_df
                else:
                    self.data_dict[new_names[count]] = new_df  
                    count+=1
        return new_df
    
    def resample_save_dates(self, name=None, new_name=None,
              resamp_meth = 'max', 
              column='Value'):
        
        """Resample basic: get one number per given timestep
        
        Only possible for month
        
        Parameters
        ----------
        name : str or None (optional)
            Select the dataframe to combine. When None, select with GUI.
            
        new_name : str or None (optional)
            New name to save it to. When None, adds 'M' to old name
            
        resamp_meth: str (optional)
            The resample method. Default value: 'max'. 
            Can also be 'min'.
            
        column : str (optional)
            The name of the column to take from the dataframe      
            Defaul value: 'Value'
        
        """
        if name is None:
            name = easygui.choicebox(msg="Pick the to resample dataframes",
                                              choices=[item for item in self.data_dict])
        
        if resamp_meth.lower() == 'max': 
            maxvalstart = -np.inf
        elif resamp_meth.lower() == 'min':
            maxvalstart = np.inf
        
        maxval = maxvalstart

        daydata = self.data_dict[name].copy()
        
        currentyearmonth = dt.datetime(daydata.iloc[0].name.year, 
                                       daydata.iloc[0].name.month, 
                                    monthrange(daydata.iloc[0].name.year, 
                                               daydata.iloc[0].name.month)[1])
        
        new_df = pd.DataFrame(columns=['month', 'Value', 'location'])
        loc = None
        for index, value in daydata.iterrows():
            val = value[column]
            
            yearmonth = dt.datetime(index.year, index.month, 
                                    monthrange(index.year, index.month)[1])
            
            if yearmonth != currentyearmonth:
                
                new_df = new_df.append({'month':currentyearmonth, 'Value': maxval,
                                        'location': loc},
                                        ignore_index=True)
                
                currentyearmonth = yearmonth
                maxval = 0
            if resamp_meth.lower() == 'max': 
                if val > maxval:
                    loc = index.date()
                    maxval = val
            elif resamp_meth.lower() == 'min':
                if val < maxval:
                    loc = index.date()
                    maxval = val
            else:
                raise ValueError("resamp_meth should either be 'min' or 'max'")
              
        if new_name is None:
            new_name = name+'M'
        
        new_df = new_df.set_index('month')
                
        self.data_dict[new_name] = new_df
        return new_df
    
    def resample_with_dates(self, subsetname,
                            datesname,
                            resamp_meth = 'sum',
                            locname = 'location',
                            daysback = dt.timedelta(days=7),
                            timeadded = dt.timedelta(hours=12),
                            lag = dt.timedelta(days=1),
                            column = 'Value',
                            drop_first=True):
        """ Resamples relatively to certain dates
        
        Parameters
        ----------
        subsetname
            The name of the DataFrame/Series with the values
        
        datesname : str 
            The name of the DataFrame with the dates
        
        resamp_meth : str (optional)
            The method to resample with. Can be 'sum' or 'mean'. 
            Default value: 'sum'
        
        locname : str (optional)
            The column name of the timestamp of the time to subset on
            Default value: 'location'
        
        daysback  : datetime.timedelta object (optional)
            Time to substract from date (how long should the mean go back)
            Default value: dt.timedelta(days=7)
            
        timeadded : datetime.timedelta object (optional)
            Time to add to date (for example when date is taken as 0:00h)
            Default value: dt.timedelta(hours=12)
                  
        lag : datetime.timedelta object (optional)
            Time not to take into account before discharge moment. 
            Positive = backwards in time. 
            Default value: dt.timedelta(days=1)
            
        column : str (optional)
            When the to subset dataset is a DataFrame instead of Series,
            the name of the column to use. Default value: 'Value'
            
        drop_first : bool (optional)
            Drop the first value when true. This is useful when first peak
            is in the beginning of the first month. 
            Default value: True
        
        Returns
        -------
        new_df : pandas.DataFrame
            DataFrame with sumsampled values
        
        """
        #TODO: annotate
        #TODO: work with gui
        maxdata = self.data_dict[datesname]#.copy()
        inputdata = self.data_dict[subsetname]#.copy()   

        new_name = None# base upon name
        if new_name is None:
            new_name = subsetname+'_subset'
            
        
        new_df = pd.Series(index = maxdata.index)
        
        if not isinstance(inputdata, pd.Series):
                inputdata = inputdata[column]
        
        
        for index, value in maxdata.iterrows():
            
            begintime = dt.datetime(value[locname].year, 
                                    value[locname].month, 
                                    value[locname].day)-daysback
        
            endtime = dt.datetime(value[locname].year, 
                                    value[locname].month, 
                                    value[locname].day)+timeadded-lag
                    
            
            selectdata = inputdata[np.logical_and(inputdata.index>begintime, 
                                                  inputdata.index<=endtime)]
            if resamp_meth.lower() == 'mean':
                new_val = np.nanmean(selectdata)
            elif resamp_meth.lower() == 'sum':
                new_val = np.nansum(selectdata)
            else:
                raise ValueError("Resamp meth can only be 'mean' or 'sum' at this point")
                
            new_df[index] = new_val
            
        if drop_first:
            new_df = new_df.drop(new_df.index[0])
            
        new_df = new_df.rename(subsetname)
        self.data_dict[new_name] = new_df 
        
        return new_df
    
    def resample_temporal_average_with_date(self,subsetname,
                                            datesname,
                                            locname='location',
                                            daysback = dt.timedelta(days=7),
                                            timeadded = dt.timedelta(hours=12),
                                            lag = dt.timedelta(days=1),
                                            column = 'NDVI',
                                            drop_first=False,
                                            datestartname = 'datestart',
                                            dateendname = 'dateend'):
        
        
        """Resamples relatively to certain dates. Input averages and returns
        averages. 
        
        Parameters
        ----------
        subsetname
            The name of the DataFrame/Series with the values
        
        datesname : str 
            The name of the DataFrame with the dates
        
        locname : str (optional)
            The column name of the timestamp of the time to subset on
            Default value: 'location'
        
        daysback  : datetime.timedelta object (optional)
            Time to substract from date (how long should the mean go back)
            Default value: dt.timedelta(days=7)
            
        timeadded : datetime.timedelta object (optional)
            Time to add to date (for example when date is taken as 0:00h)
            Default value: dt.timedelta(hours=12)
            
        lag : datetime.timedelta object (optional)
            Time not to take into account before discharge moment. 
            Positive = backwards in time. 
            Default value: dt.timedelta(days=1)
            
        column : str (optional)
            When the to subset dataset is a DataFrame instead of Series,
            the name of the column to use. Default value: 'Value'
            
        drop_first : bool (optional)
            Drop the first value when true. This is useful when first peak
            is in the beginning of the first month. 
            Default value: False
        
        Returns
        -------
        new_df : pandas.DataFrame
            DataFrame with sumsampled values
        
        """
        
        
        maxdata = self.data_dict[datesname]#.copy()
        inputdata = self.data_dict[subsetname]#.copy()   

        new_name = None# base upon name
        if new_name is None:
            new_name = subsetname+'_subset'
            
        
        new_df = pd.Series(index = maxdata.index)
                   
        for index, value in maxdata.iterrows():
            
            begintime = dt.datetime(value[locname].year, 
                                            value[locname].month, 
                                            value[locname].day)-daysback
                
            endtime = dt.datetime(value[locname].year, 
                                            value[locname].month, 
                                            value[locname].day)+timeadded-lag
                                  
            
        
            selectdata1 =  inputdata[np.logical_and(inputdata[datestartname]>=begintime,
                                                    inputdata[datestartname]<endtime)]
            
            
            #1
               
            if len(selectdata1)==1:
                factor1 = min(endtime-selectdata1[datestartname].iloc[0], 
                              selectdata1[dateendname].iloc[0]-selectdata1[datestartname].iloc[0])
                factor1 = factor1.days+factor1.seconds/86400
                val1 = selectdata1[column].iloc[0]
            else:
                factor1 = 0
                val1 = 0
                
            #2
            
            selectdata2 = inputdata[np.logical_and(inputdata[dateendname]+dt.timedelta(days=1)>begintime,
                                                    inputdata[dateendname]+dt.timedelta(days=1)<endtime)]
            
            if len(selectdata2)==1:
                factor2 = min(selectdata2[dateendname].iloc[0]+dt.timedelta(days=1)-begintime, 
                              selectdata2[dateendname].iloc[0]+dt.timedelta(days=1)-selectdata2[datestartname].iloc[0])
                factor2 = factor2.days+factor2.seconds/86400
                val2 = selectdata2[column].iloc[0]
            else:
                factor2 = 0
                val2 = 0
            
            #3
            
            selectdata3 = inputdata[np.logical_and(inputdata[datestartname]<begintime,
                                                    inputdata[dateendname]+dt.timedelta(days=1)>endtime)]
            
            if len(selectdata3)==1:
                factor3 = endtime-begintime
                factor3 = factor3.days+factor3.seconds/86400
                val3 = selectdata3[column].iloc[0]
            else:
                factor3 = 0
                val3 = 0
            
            factorios = np.array([factor1, factor2, factor3])
            sumfactors = factor1+factor2+factor3
            
            if sumfactors < 0.5:
                val = np.nan
            else:
                val = (factor1*val1+factor2*val2+factor3*val3)/sumfactors
                
            new_df[index] = val
            
        if drop_first:
            new_df = new_df.drop(new_df.index[0])
            
        new_df = new_df.rename(subsetname)
        self.data_dict[new_name] = new_df 
        return new_df
    
    def merge(self, name_list=None, timestep = 'month', 
              type_data = 'mean',
              combination='inner', export_name='merged'):
        """
        """

        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick the to combine dataframes",
                                              choices=[item for item in self.data_dict])
        
        #TODO: merge should also work with pd.Series 
        """
        to_merge_dfs = [self.data_dict[name] for name in name_list]
        
        new_df = reduce(lambda left,right: pd.merge(left, right, how='inner', 
                                                    left_index=True, 
                                                    right_index=True), 
            to_merge_dfs)
        """    
        new_df = self.data_dict[name_list[0]]
        for i in range(1,len(name_list)):
            new_df = new_df.merge(self.data_dict[name_list[i]],
                                  how = 'inner', 
                                  left_index=True, 
                                  right_index=True) 
            #print(self.data_dict[name_list[i]])
        
        #new_df = new_df.dropna(axis=0, how='any')
        
        self.data_dict[export_name] = new_df
        return new_df

    def spatially_combine_data(self, new_name, 
                               lambdas, 
                               column = None, 
                               name_list=None):
        """ Combine multiple spatial data sources with factors
        
        Alphabetically order the weights!
        
        Parameters
        ----------
        new_name : str
            Name for the new dataframe
        
        lambdas : List or numpy.Array of floats
            The weights
        
        column : str (optional)
            Which column to combine.
            !! When None, it can cause errors when certain columns are
            not float.
            
        name_list : List
            Which dataframes to combine. If None, use gui to select.
        
        Returns
        -------
        new_df : pandas.DataFrame
            The combined dataframe
        """
        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick the to combine dataframes",
                                              choices=[item for item in self.data_dict])
            

        name_list.sort()
        if column is not None:
            df_list = [self.data_dict[name_list[i]][column]*lambdas[i] for i in range(len(name_list))]
        else:
            df_list = [self.data_dict[name_list[i]]*lambdas[i] for i in range(len(name_list))]
        new_df = sum(df_list)
        
        self.data_dict[new_name] = new_df
        return new_df        
    

    
            
    def to_array(self, name=None):
        """ Return a certain DataFrame as an array
        
        Parameters
        ----------
        name : str (optional)
            Name of the DataFrame to return as numpy.ndarray
            
        Returns
        -------
        array_ : numpy.ndarray
            The numpy.ndarray of the data
        """
        data_dict = self.data_dict
        if name is None:
            name = easygui.choicebox(msg="Pick the dataframe to return as an array",
                                              choices=[item for item in self.data_dict])
            
        data = data_dict[name]
        array_ = data.to_numpy()
        
        return array_
    

    
    def to_csv(self, name_list=None, index=False):
        """ Save one or multiple dataframes to csv
        with a gui (when no names are passed)
        
        Parameters
        ----------
        name_list : List of strings (optional)
            Names of the dataframes you want to save
            
        index : bool (optional)
            Default value: False
            Save the index as well when True.
        """
        
        data_dict = self.data_dict
        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick the to save dataframes",
                                              choices=[item for item in self.data_dict])
        
        for name in name_list:
            loc = easygui.filesavebox(msg="Save the file",
                                      title="Save location",
                                      default='data.csv', filetypes=['csv'])
            data = data_dict[name]
            data.to_csv(loc, index=index)
            
            
    def load_processed_file(self, path=None, name=None, **kwargs):
        """ Load data that already has been 
        preproecessed and processed, to a DataFrame.
        
        Parameters
        ----------
        path : string or None (optional)
            The location of the file. If None, loads
            with a gui.
        
        **kwargs
            Keyword arguments of pd.read_csv
            
        Returns
        -------
        new_data : pandas.DataFrame
            The loaded DataFrame, to check whether it loaded correctly.
        """
        if path is None:
            path = easygui.fileopenbox(filetypes=['csv'])
            
        if name is None:  
            name = easygui.enterbox(msg="Enter name of data", title="Enter name",
                                    default="Data name")
        
        new_data = pd.read_csv(path, **kwargs)
        
        self.data_dict[name] = new_data
        return new_data
            

        
class WatBal:
    
    """Collect all the water balance factors and calculate balance, plot etc.
    
    Make sure the data is already resampled.
    
    Attributes
    ----------
    data : pandas.Dataframe
        The data of the water balance
        
    positive : dictionary
        For every column, whether it should be added or substracted
        True = positive
        
    unit : str
        The unit that is used
    
    area : float
        The area of the catchment
    
    Methods
    -------
    add_discharge(data, unit='m3', name='Discharge')
        Add discharge to the DataFrame
  
    add_evap(data, unit='mm', name='Evaporation')
        Add evaporation to the DataFrame
  
    add_extra(data, positive=False, name='Extra')
        Add an extra influence to the DataFrame

    add_gw(data, n=0.25, level=True, unit='cm', name='Groundwater difference')
        Add groundwater to the DataFrame
  
    add_prec(data, unit='mm', name='Precipitation')
        Add precipitation to the DataFrame
  
    add_wwtp(amount, unit='mm', name='WWTP')
        Add an uniform forcing: same per month
  
    calculate_balance(per_month=False, rel_axis='q_in')
        Calculate the water balance
  
    create_average_year()
      Average per month
  
    height(unit='mm')
        Turn the unit into 1d
  
    load_csv(path=None)
        Load csv as data. Overwrites other dataframe.
  
    minimize()
        Remove columns with NaN
  
    plot(avg_year_plt=True, title='Water balance', save_fig=False, save_path='Water Balance.png')
        Plot
  
    remove_force(name)
        Drop a force from the dataframe
  
    save_positive(path=None)
        Save dataframe to csv.
  
    to_csv(path=None)
        Save dataframe to csv.
  
    volume()
        Turn the unit into volume
    
    """
    
    
    def __init__(self, area):
        self.data = pd.DataFrame()
        self.positive = {}
        self.unit = "mm"
        self.area = area
        
    
    def add_gw(self, data, n=0.25, level=True,
               unit = "cm",
               name = 'Groundwater difference'):
        """Add groundwater to the DataFrame
        
        Parameters
        ----------
        data : pandas.Series
            Series object with the data
        
        n : float (optional)
            Default value = 0.25
            Effective porosity
            
        level : bool (optional)
            Whether or not a water level is given, in stead of a difference.
            
        unit : str (optional)
            The unit of the data. Default value: "cm"
            Can also be "m" or "mm".
            
        name : str (optional)
            The name of this data.
            Default value: "Groundwater difference"
        """

        gw = pd.Series(0, index=data.index)

        if level:
            for i in range(1, len(data)):
                gw.iloc[i] = (data.iloc[i]-data.iloc[i-1])
        else:
            gw = data
            
        if unit.lower() == "cm":
            factor = 10
        elif unit.lower() == "mm":
            factor = 1
        elif unit.lower() == "m":
            factor = 1000
        else:
            return ValueError("Unit should be either 'mm', 'cm' or 'm'")
        
        #plt.plot(gw)
        self.data[name] = gw*n*factor
        self.positive[name] = False
        
        
    def add_prec(self, data, unit="mm", name = "Precipitation"):
        """Add precipitation to the DataFrame
        
        Parameters
        ----------
        data : pandas.Series
            Series object with the data
        
        unit : str (optional)
            The unit of the data. Default value: "mm"
            Can also be "m" or "cm".
            
        name : str (optional)
            The name of this data.
            Default value: "Precipitation"
        
        """
        if unit.lower() == "mm":
            factor = 1
        elif unit.lower() == "m":
            factor = 1000
        elif unit.lower() == "cm":
            factor = 10
        else:
            return ValueError("Unit should be either 'mm', 'cm' or 'm'")
        self.data[name] = data*factor
        self.positive[name] = True
    
    def add_pot_evap(self, data, unit="mm", name = "Potential evaporation"):
        """Add evaporation to the DataFrame
        
        Parameters
        ----------
        data : pandas.Series
            Series object with the data
        
        unit : str (optional)
            The unit of the data. Default value: "mm"
            Can also be "m" or "cm".
        
        name : str (optional)
            The name of this data.
            Default value: "Evaporation"
        """
        if unit.lower() == "mm":
            factor = 1
        elif unit.lower() == "m":
            factor = 1000
        elif unit.lower() == "cm":
            factor = 10
        else:
            return ValueError("Unit should be either 'mm', 'cm' or 'm'")
        self.data[name] = data*factor
        self.positive[name] = False
    
    def add_extra(self, data, positive=False, name = 'Extra'):
        """Add an extra influence to the DataFrame
        
        Parameters
        ----------
        data : pandas.Series
            Series object with the data
            
        name : str (optional)
            The name of this data.
            Default value: "Extra"
        """
        self.data[name] = data
        self.positive[name] = positive
    
    def add_wwtp(self, amount, unit="mm", name = "WWTP"):
        """Add an uniform forcing: same per month
        
        Parameters
        ----------
        amount : float
            The WWTP amount per month
            
        unit : str (optional)
            The unit of the data. Default value: "mm"
            Can also be "m" or "m3"
            
        name : str (optional)
            The name of this data.
            Default value: "WWTP"
        """
        if unit == "mm":
            factor = 1
        elif unit == "m":
            factor = 1000
        elif unit == "m3":
            factor = 1000/self.area
        self.data[name] = amount*factor
        self.positive[name] = True
        
    
    def add_discharge(self, data, unit="m3", name = "Discharge"):
        """
        Parameters
        ----------
        unit : str (optional)
            The unit of the data. Default value: "mm"
            Can also be "m" or "m3"
        
        name : str (optional)
            The name of this data.
            Default value: "Discharge"
        
        """
        
        if unit == "mm":
            factor = 1
        elif unit == "m":
            factor = 1000
        elif unit == "m3":
            factor = 1000/self.area
        
        self.data[name] = data*factor
        self.positive[name] = False
    
    
    def remove_force(self, name):
        
        """Drop a force from the dataframe
        
        Parameters
        ----------
        name : str
            The name of the item to be removed.
        """
        self.data = self.data.drop(columns=[name])
        del self.positive[name]
        
    def volume(self):
        """Turn the unit into volume
        
        """
        area = self.area
        if self.unit.lower() == "mm":
            factor = area/1000.
        elif self.unit.lower() == "m":
            factor = area
        else:
            raise ValueError("self.unit should be either 'mm' or 'm'")
        
        self.data.multiply(factor)
        self.unit = "m3"
    
    def height(self,unit="mm"):
        """Turn the unit into 1d
        
        Parameters
        ----------          
        unit : str (optional)
            Default value: "mm"
            The unit to convert to.
        """
        area = self.area
        if unit.lower() == "mm":
            factor = area/1000.
        elif unit.lower() == "m":
            factor = area
        else:
            raise ValueError("unit should be either 'mm' or 'm'")
        
        self.data.multiply(1/factor)
        self.unit = unit
        
    def to_csv(self, path=None):
        """Save dataframe to csv.
        
        Parameters
        ----------
        path : str (optional)
            The path of the file
        
        """
        
        if path is None:
            path = easygui.filesavebox(msg="Save the df to a location",
                                       title="Save",
                                       default="Water_balance.csv",
                                       filetypes=["csv"])
        self.data.to_csv(path)
        print("Don't forget to also save the positive dict")
        
    def save_positive(self, path=None):
        """Save dataframe to csv.
        
        Parameters
        ----------
        path : str (optional)
            The path of the file
        
        """
                
        if path is None:
            path = easygui.filesavebox(msg="Save to JSON file", 
                                title="Save", default="data_sources.json",
                                filetypes=["json"])
        with open(path, 'w') as fp:
            json.dump(self.positive, fp)
        print("Don't forget to also save the data")
        
    def load_csv(self, path=None):
         """Load csv as data. Overwrites other dataframe.
         
         Parameters
         ----------
         path : str (optional)
             The path of the file
         """
         
         if path is None:
             path = easygui.fileopenbox(msg="Load the df as csv",
                                        title="Load",
                                        filetypes=["csv"])
             
         self.data = pd.read_csv(path)
         self.data.index =  pd.to_datetime(self.data.index)
         
         
    def plot(self, avg_year_plt = True,
             title = "Water balance",
             save_fig = False,
             save_path = "Water Balance.png"):
        """Plot the water balance 
        
        """
        plt.figure(figsize=(14,7))
        
        if avg_year_plt:
            month_list = ["Dec1", "Jan", "Feb", 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', "Jan1"]
            for src in self.avg_year:
                values = np.zeros(14)
                values[1:-1] = self.avg_year[src]
                values[0] = values[-2]
                values[-1] = values[1]
                plt.plot(month_list, values, label=src, marker = '|', linewidth = 2)
                plt.xlabel("Month",
                       color = '#4d4d4d')
                plt.xlim(0.5,12.5)
                #plt.ylim((-55,140))
            
            try:
                plt.bar(month_list[1:-1], self.diff, 
                        label = 'Balance error', alpha = 0.7, 
                        color = '#dc6f6f')
            except:
                pass
        
        else:
            self.data.plot(figsize=(12,8))
            plt.xlabel("Date")
        
        plt.title(title,
                       fontsize=18)
        plt.legend( bbox_to_anchor= (1.04, 1), loc = 'upper left')        
        plt.ylabel(f"Water quantity ({self.unit})",
                       color = '#4d4d4d')
        plt.grid(axis='y', linestyle = ':', linewidth=1)
        plt.xticks(
                       color = '#4d4d4d')
        plt.yticks(
                       color = '#4d4d4d')
        plt.axhline(0, color = '#4d4d4d')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_path, dpi=200)
            
    def plot_pot_evap_prec(self,
             title = "Precipitation and evaporation",
             save_fig = False,
             save_path = "Precipitation and evaporation.png"):
        """Plot the water balance with only evaporation and precipiation
        
        Avg_year should have been made
        """
        plt.figure(figsize=(9,6))
        
        month_list = ['Dec1', "Jan", "Feb", 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan1']
        
        pot_evap = np.zeros(14)
        pot_evap[1:-1] = self.avg_year["Potential evaporation"]
        pot_evap[0] = pot_evap[-2]
        pot_evap[-1] = pot_evap[1]
        
        prec = np.zeros(14)
        prec[1:-1] = self.avg_year["Precipitation"]
        prec[0] = prec[-2]
        prec[-1] = prec[1]
        
        
        plt.plot(month_list, pot_evap, 
                 linewidth=1.7,
                 label="Potential evaporation", 
                 color='maroon',
                 marker = '|',
                 markersize=10,
                 markeredgewidth = 1.7)
        plt.plot(month_list, prec,
                 linewidth=1.7,
                 label="Precipitation", 
                 color='midnightblue',
                 marker = '|',
                 markersize = 10,
                 markeredgewidth = 1.7)
        plt.xlabel("Month",
                       color = '#4d4d4d')
        
        plt.fill_between(month_list, pot_evap, 
                         prec,
                         where= pot_evap >=  prec, 
                         interpolate=True,
                         facecolor='orangered',
                         label = 'Groundwater depletion')
        
        plt.fill_between(month_list, pot_evap, 
                         prec,
                         where= pot_evap <=  prec, 
                         interpolate=True,
                         facecolor= 'lightseagreen',
                         label= 'Groundwater recharge')
        
        plt.title(title,fontsize = 14)
        plt.legend()        
        plt.ylabel(f"Water flux ({self.unit})",
                       color = '#4d4d4d')
        plt.grid(axis='y')
        plt.xlim(0.5,12.5)
        plt.xticks(color = '#4d4d4d')
        plt.yticks(color = '#4d4d4d')
    
        
        if save_fig:
            plt.savefig(save_path, dpi=200)
        
    def minimize(self):
        """Remove columns with NaN
        
        """
        self.data = self.data.dropna()
        self.is_minimized = True
        
    def _check_minimized(self):
        
        if not self.is_minimized:
            raise NameError("DataFrame not yet minimized")
    
    def calculate_balance(self, per_month = False, rel_axis='q_in'):
        """
        
        Parameters
        ----------
        per_month : bool (optional)
            Water balance per month.
            Default value: False
            
        rel_axis : str (optional)
            Default value: 'q_in'.
            To what parameter the difference is compared.
            Either 'q_in', 'q_out' or 'del_s'.
        
        Returns
        -------
        diff : float
            Difference water balance
        
        rel_diff : float
            Relative difference water balance
        """
        self._check_minimized()
        
        q_in = 0
        q_out = 0
        
        if per_month:
            diff = np.zeros(12)
            for src in self.positive:
                if self.positive[src]:
                    diff = diff + self.avg_year[src]
                else:
                    diff = diff - self.avg_year[src]   
        else:
            for src in self.positive:
                sumdata = np.sum(self.data[src])
                if self.positive[src]:
                    q_in += sumdata
                else:
                    q_out += sumdata
            
        if per_month:
            plt.plot(diff)  
            self.diff = diff
            return diff
            
        
            
        if not per_month:
            diff = q_in-q_out
            if rel_axis.lower() == 'q_in':
                rel_diff = diff/q_in
            elif rel_axis.lower() == 'q_out':
                rel_diff = diff/q_out
            elif rel_axis.lower() == 'del_s':
                del_s = np.sum(self.data["Groundwater difference"])
                rel_diff = diff/del_s
            else:
                raise ValueError("[!] rel_axis should be either 'q_in', 'q_out' or 'del_s'")
                
            return diff, rel_diff
        
        
    def create_average_year(self):
        """ Average per month
        """   
        avg_year = {}
        
        for src in self.data.columns:
            avg_year[src] = np.zeros(12)
            for i in range(12):
                avg_year[src][i] = np.mean(self.data[src][self.data.index.month==i+1].values)
        self.avg_year = avg_year
        
    def budyko_curve(self, plot = True, title="Budyko curve", max_x=1.8,
                     save_figure=False, extra_sources=None,
                     save_path="../figures/budyko curve.png"):
        """
        Actual evaporation based on the water balance
        
        Parameters
        ----------
        plot : bool (optional)
            Default value: True
            Plot a figure or not
            
        title : str (optional)
            Default value: "Budyko curve"
            The title of the plot
            
        max_x : float (optional)
            Length of the plot. 
            Default value: 2
            
        Returns
        -------
        arid_index : float
            Aridity index
            
        evap_index : float
            Evaporation index
        """
        print("Budyko curve incoming!")
        self._check_minimized()
            
        pot_evap = np.sum(self.data["Potential evaporation"])
        prec = np.sum(self.data["Precipitation"])
        xval = pot_evap/prec
        
        bal = self.calculate_balance()[0]
        
        yval = (pot_evap+bal)/prec 
        
        
        if plot:
            plt.figure(figsize=(12.3,8))
            
            def simpleaxis(ax):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
    
            ax = plt.subplot(111)
            simpleaxis(ax)
            # Plot the limits
            plt.plot([0,1],[0,1], 'darkgrey')
            plt.plot([1,max_x],[1,1], 'darkgrey')
            plt.text((max_x-1)/2+0.9, 1.02, "Water limit")
            plt.text(0.45, 0.66, "Energy limit", rotation=45)
            plt.axis('scaled')
            plt.xlim([0,max_x])
            plt.ylim([0,1.15])
            plt.xlabel(r"Aridity index "+r"($E_p/P$)",
                       color = '#4d4d4d')
            plt.ylabel(r"Evaporation index "+ r"($E_a/P$)",
                       color = '#4d4d4d')
            plt.title(title, fontsize = 14)
            
            plt.xticks(color = '#4d4d4d')
            plt.yticks(color = '#4d4d4d')
            

            plt.scatter([xval], [yval], marker="D")
            plt.text(xval+0.02, yval-0.04, "Catchment")
            plt.grid(which='major', axis='both', color='darkgrey',
             linestyle='dashdot', linewidth=0.3)
            
            if save_figure:
                plt.savefig(save_path, dpi=200)
        
        arid_index = xval
        evap_index = yval
        
        return arid_index, evap_index


def water_balance(self, disch_all,
                  rain_all,
                  evap_all,
                  gw_all,
                  rwzi_single_month = 0,
                  average_year=False, 
                  plot_figure=False,
                  data_unit="mm/month",
                  save_figure=False,
                  save_path="../figures/overall_water_balance.png"):
    """
    Return and plot the water balance
    
    ! When average_year==True, make shure that all items have 
    the same timeframe. 
    
    Parameters
    ----------
    disch_all : numpy.array or pandas.DataFrame
        The discharge measurements
    
    rain_all : numpy.array or pandas.DataFrame
        The precipiation measurements
        
    evap_all : numpy.array or pandas.DataFrame
        The evaporation data
    
    gw_all : numpy.array or pandas.DataFrame
        The groundwater DIFFERENCE data
        
    rwzi_single_month : float
        Discharge of the WWTP (RWZI in Dutch) in a single month.
        Default value: 0.
        
    average_year : bool
        When True, averages over every January, February etc.
        Default value: False
        
    plot_figure : bool
        Plot the figure.
        Default value: False
        
    data_unit : str
        The unit of the data
        Default value: "mm/month"
        
    save_figure : bool
        Whether to save the figure
        Default value: False
        
    save_path : str
        Save the figure to this path.
    
    Returns
    -------
    total_diff : float
        The total water balance difference
        
    rel_total_diff : float
        The total water balance difference divided by the sum of the discharge
        
    diff : numpy.array or pandas.DataFrame
        The difference per timestep
        
    rel_diff : numpy.array or pandas.DataFrame
        The relative difference per timestep
    """
    #TODO: make fill_between
    
    if average_year:
        
        disch_month = np.zeros(12)
        rain_months = np.zeros(12)
        evap_month= np.zeros(12)
        gw_month = np.zeros(12)
        for i in range(12):
            disch_month[i] = np.mean(disch_month[disch_month.index.month==i+1].values)
            rain_months[i] = np.mean(rain_all[rain_all.index.month==i+1].values)
            evap_month[i] = np.mean(evap_month[evap_month.index.month==i+1].values)
            gw_month[i] = np.mean(gw_month[gw_month.index.month==i+1].values)
        
        rwzi=np.ones(12)*rwzi_single_month
        extr_wat = rain_months-evap_month+rwzi-gw_month
        diff = disch_month - extr_wat
        rel_diff = diff/disch_month
        total_diff = np.sum(disch_month) - np.sum(extr_wat)
        rel_total_diff = total_diff/np.sum(disch_month)
        
        if plot_figure:
            
            plt.figure(figsize=(12,8))
            plt.plot(range(1,13),evap_month, label='Evaporation')
            plt.grid(axis='y')
            plt.plot(range(1,13),rain_months, label='Precipitation')
            plt.plot(range(1,13), disch_month, label='Discharge')
            plt.plot(range(1,13), rwzi, label='WWTP')
            plt.plot(range(1,13), gw_month,'y', label="Groundwater difference")
            
            plt.plot(range(1,13), extr_wat, 'k', label='Surplus water')
            
            
            month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            print(extr_wat-disch_month)
            plt.xticks(range(1,13), month_list)
            plt.ylim(-37,110)
            plt.ylabel(data_unit)
            plt.xlabel('month')
            plt.title("Water Balance per month - Average year")
            plt.xlim(0.9,12.1)
            plt.legend()
            if save_figure:
                plt.savefig(save_path, dpi=200)

    
    else:
        rwzi_all = rain_all.copy()
        rwzi_all[:] = rwzi_single_month
        extr_wat_all = rain_all-evap_all+rwzi_all-gw_all
        
        total_diff = np.sum(disch_all)-np.sum(extr_wat_all)
        rel_total_diff = total_diff/np.sum(disch_all)
        diff = disch_all-extr_wat_all
        rel_diff = diff/disch_all
        
        if plot_figure:
            plt.figure(figsize=(12,8))
            plt.plot(disch_all, label= "Discharge")
            plt.plot(extr_wat_all, label= "Surplus water")
            
            #plt.plot(gw_all, label="GW")
            plt.legend()
            plt.xlim(dt.datetime(2011,1,1))
            plt.xlabel('Date')
            plt.ylabel(data_unit)
            plt.title("Discharge vs surplus water of the water balance")
            plt.grid(axis='y')
            if save_figure:
                plt.savefig(save_path, dpi=200)

    return total_diff, rel_total_diff, diff, rel_diff


def back_factors(n=7, power=0.5): #keep it simple or ask Markus because I do not have a reason right now
    """Calculate the factors 
    
    Parameters
    ----------
    n : int (optional)
        Default value: 7
        The number of days to take into account.
        
    power : float (optional)
        Default value: 0.5
        The power in the formula
        
    Returns
    -------
    factors : numpy.array of floats
        The factors of the values. Sum of the factors = 1
    
    
    """
    factors_pre = np.arange(1,n+1)**power
    factors = factors_pre/np.sum(factors_pre) 
    return factors


def select_date():
    date = None
    return date


def back_values(data, date, n=7, power=0.5):
    
    pass





