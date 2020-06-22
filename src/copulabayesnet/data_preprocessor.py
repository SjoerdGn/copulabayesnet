# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:11:19 2019

Filtering, testing water balance 

@author: Sjoerd Gnodde
"""
import easygui
import pandas as pd
import numpy as np
#import datetime as dt
import json
from scipy.stats import zscore

class DataPreProc:
    
    """
    Preprocess the data
    
    ...
    
    
    Attributes
    ----------
    data_dict : dictionary with pandas.DataFrames with the data
        Formatted as:
        name data : dataframe
        
    source : dictonary 
        The dictionary with the source of the data:
        name data : path file
    
    Methods
    -------
    basic_filter(zval=1.5, name_list=None, inplace=True)
       Do a basic, general filter over the given data. 
       Method is the Standard score: 
               https://en.wikipedia.org/wiki/Standard_score 
               (also called z-score)
       
       When single column, removes rows.
       When multiple columns, turns into NaN. 
       
       Does not return anything, but saves it to object.
       
    combine_knmi_decades(new_name=None, name_list=None)
       Combine multiple decades of KNMI data 
    
    load_from_gui()
       Load a data source from a basic GUI
    
    load_from_name_and_path(path, name, data_type='dtype2', parameters=None)
       Load a file with a given path and name. Saves it in object
 
    load_multiple_from_gui(n)
     
     
    load_source(path=None)
       Load the data source from the documents
 
    save_source(path=None)
       Save the data source to a file given in the box
    
    """
    
    
    def __init__(self):
        self.data_dict = {}
        self.source = {}


    def load_from_gui(self):
        """Load a data source from a basic GUI
        """
        path = easygui.fileopenbox()
        name = easygui.enterbox(msg="Enter name of data", title="Enter name",
                                default="Water level")
        
        data_type = easygui.buttonbox(msg="Choose a data type",
                                      title="Data type",
                                      choices=("dtype2", "KNMI_hourly", "KNMI_daily"))
        self.load_from_name_and_path(path, name, data_type=data_type)
        

    def load_from_name_and_path(self, path, name, data_type='dtype2', parameters=None,
                                lentest = False,
                                lenmin = 10):
        """ Load a file with a given path and name. Saves it in object
        
        
        Parameters
        ----------
        path : str
            Path of the file
            
        name : str
            Give the file a usefule name
            
        parameters : list of str or None (optional)
            What parameters to load. When None, loads all.
            
        Returns 
        -------
        new_data : pandas.DataFrame
            KNMI data in pandas dataframe
        
        
        
        """
        if data_type.lower() == 'dtype2':
            new_data = self._get_dtype2_file(path)
        elif data_type.lower() == 'knmi_hourly':
            print(path)
            new_data = self._get_knmi_file(path, parameters)
        elif data_type.lower() == 'knmi_daily':
            new_data = self._get_knmi_daily_file(path)
        
        add = True
        if lentest:
            if len(new_data) < lenmin:
                add = False
                
        if add:   
            if name not in self.data_dict:
                self.data_dict[name] = new_data
                self.source[name] = [path, data_type] # to be able to load 
            
            else:
                self.data_dict[name+'_'] = new_data
                self.source[name+'_'] = [path, data_type] # to be able to load 
                print("There is already data with this name.")
            
        else:
            print("Data was too short")
      
    
    def _get_knmi_daily_file(self, path):
        """Load daily KNMI data, now only for evaporation
        
        Parameters
        ----------
        path : str
            Path to the file
        
        Returns
        -------
        new_data : pandas.DataFrame
            Dataframe with the daily data
        """
        evap_data = pd.read_csv(path, 
            sep=',',
            skipinitialspace=True,
            skiprows=10, escapechar='#',  dtype={'YYYYMMDD':str})

        evap_data = evap_data.iloc[1:]
        evap_data = evap_data.set_index('YYYYMMDD')
        evap_data.index = pd.to_datetime(evap_data.index)
        try: 
            evap_data.drop(columns=['STN'])
        except:
            pass
        new_data = evap_data
        
        return new_data
    
    def _get_dtype2_file(self, path):
        """Load a .csv file from dtype2
        
        Parameters
        ----------
        path : str
            Path of the file
            
        Returns 
        -------
        new_data : pandas.DataFrame
            KNMI data in pandas dataframe
        """
        new_data = pd.read_csv(path, sep = ';',decimal=',', dtype={'Value [m3/s]':float, 'Value':float})
        
        try:
                new_data['Timestamp'] = pd.to_datetime(new_data['Timestamp'], format = "%Y-%m-%d %H:%M:%S+01:00")
                new_data = new_data.set_index('Timestamp', drop=True)
            
        except:
            try:
                new_data['Date/Time'] = pd.to_datetime(new_data['Date/Time'], format = "%d-%m-%Y %H:%M:%S")
                new_data = new_data.set_index('Date/Time', drop=True)
            except:
                pass
        
        return new_data
            
      
            
    # def _get_knmi_file(self, path, parameters):
    #     """Load KNMI data with readKNMI.readKNMMIuurgeg
        
    #     Parameters
    #     ----------
    #     path : str
    #         Path of the file
            
    #     parameters : list of str or None (optional)
    #         What parameters to load. When None, loads all.
            
    #     Returns 
    #     -------
    #     new_data : pandas.DataFrame
    #         KNMI data in pandas dataframe
    #     """
    #     if parameters is None:
    #          parameters = ["DD",   "FH",   "FF",   "FX",    "T",  
    #                        "T10",   "TD",   "SQ",    "Q",   "DR",   
    #                        "RH",    "P",   "VV",    "N",    "U",   
    #                        "WW",   "IX",    "M",    "R",    "S",    
    #                        "O",    "Y"]
    #     print(path)
    #     new_data = readKNMI.readKNMIuurgeg(path, parameters)
    #     return new_data
    
    def combine_knmi_decades(self, new_name=None, name_list=None):
        """Combine multiple decades of KNMI data 
        
        Parameters
        ----------
        new_name: str
            The name of the new frame
            
        name_list : list of strings (optional)
            Select the dataframes to combine
            
        Returns
        -------
        new_data : pandas.DataFrame
            dataframe with the combined KNMI data
        
        """
        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick the to combine dataframes",
                                              choices=[item for item in self.source])
            
        if new_name is None:
            new_name = easygui.enterbox(msg="Enter name of the new frame", 
                                        title="Enter name",
                                        default="KNMI")
            
        combine_items = [self.data_dict[name] for name in name_list]
        new_data = pd.concat(combine_items, ignore_index=False, sort=True)
        self.data_dict[new_name] = new_data
        return new_data
    

        
    def load_multiple_from_gui(self, n):
        for _ in range(n):
            self.load_from_gui()
            

    
    def save_source(self, path=None):
        """ Save the data source to a file given in the box
        
        Parameters
        ----------
        path : str or None (optional)
            path of file to save. If None, asks for path with box
            
        """
        source = self.source
        if path is None:
            path = easygui.filesavebox(msg="Save to JSON file", 
                                title="Save", default="data_sources.json",
                                filetypes=["json"])
        with open(path, 'w') as fp:
            json.dump(source, fp)
        
    def load_source(self, path=None):
        """ Load the data source from the documents
        
        Parameters
        ----------
        path : str or None (optional)
            path of file to load. If None, asks for path with box
            
        """
        if path is None:
            path = easygui.fileopenbox(msg="Load json with file source",
                                            title="Load",
                                            filetypes=["json"])
        
        with open(path) as json_file:
                source = json.load(json_file)
                
        self.source = source
        
        for item in source:
            self.load_from_name_and_path(source[item][0], item, 
                                         data_type=source[item][1])
        
        

    
    def basic_filter(self, zval=1.5, 
                     name_list=None, 
                     inplace=True):
        """Do a basic, general filter over the given data. 
        Method is the Standard score: 
                https://en.wikipedia.org/wiki/Standard_score 
                (also called z-score)
        
        When single column, removes rows.
        When multiple columns, turns into NaN. 
        
        Does not return anything, but saves it to object.
        
        Parameters
        ----------
        zval : float (optional)
            Default The maximum z-score of items. Higher than this value will
            be removed
            
        names : list of str (optional)
            The items you want to filter 
        
        inplace : bool (optional)
            Default value: True. When True, replaces old data.
            Else, saves with new name; with 'F' appended to old name.
        
        
        """
        data_dict = self.data_dict
        
        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick to filter dataframes",
                                choices=[item for item in self.data_dict])
        
        for name in name_list:
            if len(data_dict[name].columns) == 1:
                #print("test")
                column = 'Value'
                #print(data_dict[name][column])
                #zscore = data_dict[name][column]
                #print(~np.isnan(data_dict[name][column]))
                #z_score[~np.isnan(data_dict[name][column])] = zscore(
                #    data_dict[name][column][~np.isnan(data_dict[name][column])])
                z_score = zscore(data_dict[name][column], nan_policy='omit')
                #print(z_score)
                z_filter = np.where(np.abs(z_score) < zval)  # or z_filter, _ = ???
                z_remove = np.where(np.abs(z_score) > zval) # or z_filter, _ = ???
                df_remove = data_dict[name][column].iloc[z_remove]
                new_df = data_dict[name].iloc[z_filter]
            
            else:
                columns = data_dict[name].columns.values
                for column in columns:
                    z_score = zscore(data_dict[name][column])
                    z_remove = np.where(np.abs(z_score) > zval) # or z_filter, _ = ???
                    data_dict[name][column].iloc[z_remove] = np.NaN
                    new_df = data_dict[name]
            
            if inplace:
                self.data_dict[name] = new_df
                
            else:
                self.data_dict[name+'F'] = new_df
        return df_remove

    def RH_filter(self, name_list=None, inplace=True):
        """
        """
        
        data_dict = self.data_dict
        
        if name_list is None:
            name_list = easygui.multchoicebox(msg="Pick to filter dataframes",
                                choices=[item for item in self.data_dict])
            
        for name in name_list:
            data_dict[name].loc[data_dict[name]['RH'] < 1., 'RH'] = 0
            new_df = data_dict[name]/10.
            
            if inplace:
                self.data_dict[name] = new_df
                
            else:
                self.data_dict[name+'F'] = new_df
            
        
            
class CorrMatrix:
    
    """
    Preprocess the data
    
    ...
    
    
    Attributes
    ----------
    data : dictionary with pandas dataframes
    
    Methods
    -------
    
    load_file()
        Load a single file into the dictionary
    
    """
    
    
    def __init__(self, loc=None):
        if loc is None:
            loc = easygui.fileopenbox(msg="Select the R matrix", 
                                      title="Open file", 
                                      filetypes=['csv', 'txt'])
        self.R = np.genfromtxt(loc, delimiter = '\t', autostrip=True)
        self.R = self.R[:,:-1]
    
        