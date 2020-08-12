# -*- coding: utf-8 -*-
"""
Created on 5-12-2019

@author: Sjoerd Gnodde
"""

import matplotlib.pyplot as plt
import matplotlib.animation as manimation

#from biokit.viz.scatter import ScatterHist # biokit is not working right now
#from biokit.viz import corrplot
import numpy as np

from pycopula.visualization import pdf_2d, cdf_2d
#from tools.bncopula import Copula2d
from matplotlib import cm

import datetime as dt
metadata = dict(title='Copula', artist='Sjoerd Gnodde',
                comment='Enjoy!')
import matplotlib


def cop_2_param(Copula2d,
                type_copula = "pdf",
                plot_method = "3d",
                zclip = 6,
                elev = 15.,
                azim = 280, 
                x_name = None,
                y_name = None, 
                title = None,                
                save_fig=False, 
                save_path='pred_target.png'):
    """Plots a 2-parameter copula, either on a 2d or 3d plane, either the pdf or the cdf.
    
    Parameters
    ----------
    Copula2d : bncopula.Copula2d Object
        A copula object
        
    type_copula : str (optional)
        Default value: "pdf"
        The capacity of the copula: either 'pdf' or 'cdf'
        
    plot_method : str (optional)
        Default value: 3d.
        Plot either 2d or 3d.
    
    zclip : float (optional)
        Default value: 6.
        The maximum z-value to plot 
        
    elev : float (optional)
        Default value: 15.
        From which elevation to watch the 3d plot
        
    azim : float (optional)
        Default value: 280.
        The angle to watch the 3d plot.
        
    x_name : str (optional)
        The name of the x-axis. When None, tries to find name in the copula.
        
    y_name : str (optional)
        The name of the y-axis. When None, tries to find name in the copula. 
        
    title : str (optional)
        Title of the figure.
        
    save_fig : bool (optional)
        Whether or not to save the figure.
        
    save_path : str (optional)
        Path where to save the figure to

    """
    
    copula = Copula2d.copula
    if type_copula.lower() == "pdf":
        level = pdf_2d(copula, zclip = zclip) 
    elif type_copula.lower() == "cdf":
        level = cdf_2d(copula)
    else:
        raise ValueError("type_copula should be either 'pdf' or 'cdf'")
    
    x,y = np.meshgrid(level[0], level[1])
    
    # Handle data names
    if x_name is None:
        try:
            x_name = Copula2d.x_name
        except:
            x_name = "X values"
    
    if y_name is None:
        try:
            y_name = Copula2d.y_name
        except:
            y_name = "Y values"
    
    fig = plt.figure(figsize = (10,8))
    
    if plot_method.lower() == "3d":
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.plot_surface(x,y,level[2],cmap=cm.coolwarm)
        if type_copula.lower() == "pdf":
            ax.set_zlabel("Probability ($p$)")
        elif type_copula.lower() == "cdf":
            ax.set_zlabel("Cumulative probability")
    elif plot_method.lower() == "2d":
        ax = fig.add_subplot(111)
        maxval = round(np.min([np.nanmax(np.array(level[2])), zclip]))
        cm_levels = np.linspace(0,maxval, np.max([20, maxval*5])+1)
        cmp = ax.contourf(x,y,level[2],cmap=cm.coolwarm, levels = cm_levels)
        ax.contour(x,y,level[2], colors='k',linewidths=0.5, levels = cm_levels)
        cbar = plt.colorbar(cmp)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis('scaled')

        if type_copula.lower() == "pdf":
            cbar.set_label("Probability (p)", rotation=270, labelpad = 20)
        elif type_copula.lower() == "cdf":
            cbar.set_label("Cumulative probability", rotation=270, labelpad = 20) 
    else:
        raise ValueError("Plot method should be either '2d' or '3d'")

    if title is None:
        ax.set_title(f"Figure of a copula ({type_copula})")
    else:
        ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
    if save_fig:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
     
    
def different_cop(x, y, level,
                type_copula = "pdf",
                plot_method = "3d",
                zclip = 6,
                elev = 15.,
                azim = 280, 
                x_name = "$u_1$",
                y_name = "$u_2$", 
                title = None,                
                save_fig=False, 
                save_path='pred_target.png'):
    """Plots any 2-parameter copula, either on a 2d or 3d plane, either the pdf or the cdf. 
    
    Parameters
    ----------
    Copula2d : bncopula.Copula2d Object
        A copula object
        
    type_copula : str (optional)
        Default value: "pdf"
        The capacity of the copula: either 'pdf' or 'cdf'
        
    plot_method : str (optional)
        Default value: 3d.
        Plot either 2d or 3d.
    
    zclip : float (optional)
        Default value: 6.
        The maximum z-value to plot 
        
    elev : float (optional)
        Default value: 15.
        From which elevation to watch the 3d plot
        
    azim : float (optional)
        Default value: 280.
        The angle to watch the 3d plot.
        
    x_name : str (optional)
        The name of the x-axis. When None, tries to find name in the copula.
        
    y_name : str (optional)
        The name of the y-axis. When None, tries to find name in the copula. 
        
    title : str (optional)
        Title of the figure.
        
    save_fig : bool (optional)
        Whether or not to save the figure.
        
    save_path : str (optional)
        Path where to save the figure to

    """
          
    fig = plt.figure(figsize = (7,5.6)) # 10,8
    
    if plot_method.lower() == "3d":
        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig) 
        ax.view_init(elev=elev, azim=azim)
        level[level > zclip] = zclip
        ax.plot_surface(x,y,level,cmap=cm.coolwarm)
        plt.tight_layout()
        #ax.set_zticklabels((),  color = '#4d4d4d', fontsize = 14)
        if type_copula.lower() == "pdf":
            ax.set_zlim(0,zclip)
            ax.set_zlabel("Probability (p)",  color = '#4d4d4d', fontsize = 14)
        elif type_copula.lower() == "cdf":
            ax.set_zlabel("Cumulative probability",  color = '#4d4d4d', fontsize = 14)
        ax.w_zaxis.set_pane_color((250/255, 250/255, 250/255))
        ax.w_yaxis.set_pane_color((225/255, 225/255, 225/255))
        ax.w_xaxis.set_pane_color((190/255, 190/255, 190/255))
        ax.xaxis._axinfo["grid"]['linestyle'] = ':'
        ax.yaxis._axinfo["grid"]['linestyle'] = ':'
        ax.zaxis._axinfo["grid"]['linestyle'] = ':'
        #ax.grid(linestyle = ':')
        #fig.set_tight_layout(True)
    elif plot_method.lower() == "2d":
        ax = fig.add_subplot(111)
        maxval = round(np.min([np.nanmax(np.array(level)), zclip]))
        cm_levels = np.linspace(0,maxval, int(np.max([20, maxval*5])+1))
        cmp = ax.contourf(x,y,level,cmap=cm.coolwarm, levels = cm_levels)
        ax.contour(x,y,level, colors='k',linewidths=0.5, levels = cm_levels)
        cbar = plt.colorbar(cmp)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis('scaled')

        if type_copula.lower() == "pdf":
            cbar.set_label("Probability (p)", rotation=270, labelpad = 20,
                            color = '#4d4d4d', fontsize = 14)
        elif type_copula.lower() == "cdf":
            cbar.set_label("Cumulative probability", rotation=270, labelpad = 20,
                            color = '#4d4d4d', fontsize = 14)
    else:
        raise ValueError("Plot method should be either '2d' or '3d'")

    if title is None:
        ax.set_title(f"Figure of a copula ({type_copula})", 
                      fontsize = 16)
    else:
        ax.set_title(title,  fontsize = 14)
    ax.set_xlabel(x_name,  color = '#4d4d4d', fontsize = 14)
    ax.set_ylabel(y_name,  color = '#4d4d4d', fontsize = 14)
    #ax.set_xticklabels((),  color = '#4d4d4d', fontsize = 14)
    ax.tick_params(axis='both', which='major',color = '#4d4d4d')
    
    if save_fig:
        print(save_path)
        plt.savefig(save_path, dpi=200)   

def different_cop_video(x, y, level,
                type_copula = "pdf",
                plot_method = "3d",
                zclip = 6,
                elev = 15.,
                azim = 280, 
                x_name = "$u_1$",
                y_name = "$u_2$", 
                title = None,  
                save_video = False,
                save_fig=False, 
                save_path='pred_target.png'):
    """Plots any 2-parameter copula, either on a 2d or 3d plane, 
    either the pdf or the cdf. 
    
    Parameters
    ----------
    Copula2d : bncopula.Copula2d Object
        A copula object
        
    type_copula : str (optional)
        Default value: "pdf"
        The capacity of the copula: either 'pdf' or 'cdf'
        
    plot_method : str (optional)
        Default value: 3d.
        Plot either 2d or 3d.
    
    zclip : float (optional)
        Default value: 6.
        The maximum z-value to plot 
        
    elev : float (optional)
        Default value: 15.
        From which elevation to watch the 3d plot
        
    azim : float (optional)
        Default value: 280.
        The angle to watch the 3d plot.
        
    x_name : str (optional)
        The name of the x-axis. When None, tries to find name in the copula.
        
    y_name : str (optional)
        The name of the y-axis. When None, tries to find name in the copula. 
        
    title : str (optional)
        Title of the figure.
        
    save_fig : bool (optional)
        Whether or not to save the figure.
        
    save_path : str (optional)
        Path where to save the figure to

    """
          
    fig = plt.figure(figsize = (7,5.6)) # 10,8
    if save_video:
        path2 = "../figures/part6_2.mp4"
        matplotlib.use("Agg")
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=10, metadata=metadata, bitrate = 2000)
        with writer.saving(fig, path2, 200):   
            
            if plot_method.lower() == "3d":
                
                    ax = fig.add_subplot(111, projection='3d')
                    if title is None:
                        ax.set_title(f"Figure of a copula ({type_copula})", 
                                      fontsize = 16)
                    else:
                        ax.set_title(title,  fontsize = 14)
                        ax.set_xlabel(x_name,  color = '#4d4d4d', fontsize = 14)
                        ax.set_ylabel(y_name,  color = '#4d4d4d', fontsize = 14)
                        #ax.set_xticklabels((),  color = '#4d4d4d', fontsize = 14)
                        ax.tick_params(axis='both', which='major',color = '#4d4d4d')

                    
                    level[level > zclip] = zclip
                    ax.plot_surface(x,y,level,cmap=cm.coolwarm)
                    #plt.tight_layout()
                    #ax.set_zticklabels((),  color = '#4d4d4d', fontsize = 14)
                    if type_copula.lower() == "pdf":
                        ax.set_zlim(0,zclip)
                        ax.set_zlabel("Probability (p)",  color = '#4d4d4d', fontsize = 14)
                    elif type_copula.lower() == "cdf":
                        ax.set_zlabel("Cumulative probability",  color = '#4d4d4d', fontsize = 14)
                    ax.w_zaxis.set_pane_color((250/255, 250/255, 250/255))
                    ax.w_yaxis.set_pane_color((225/255, 225/255, 225/255))
                    ax.w_xaxis.set_pane_color((190/255, 190/255, 190/255))
                    ax.xaxis._axinfo["grid"]['linestyle'] = ':'
                    ax.yaxis._axinfo["grid"]['linestyle'] = ':'
                    ax.zaxis._axinfo["grid"]['linestyle'] = ':'
                    for i in np.linspace(0,360,100):
                        ax.view_init(elev=25+i/8, azim=i)
                        writer.grab_frame()
                #ax.grid(linestyle = ':')
                #fig.set_tight_layout(True)
                
                
            elif plot_method.lower() == "2d":
                ax = fig.add_subplot(111)
                maxval = round(np.min([np.nanmax(np.array(level)), zclip]))
                cm_levels = np.linspace(0,maxval, int(np.max([20, maxval*5])+1))
                cmp = ax.contourf(x,y,level,cmap=cm.coolwarm, levels = cm_levels)
                ax.contour(x,y,level, colors='k',linewidths=0.5, levels = cm_levels)
                cbar = plt.colorbar(cmp)
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                ax.axis('scaled')
        
                if type_copula.lower() == "pdf":
                    cbar.set_label("Probability (p)", rotation=270, labelpad = 20,
                                    color = '#4d4d4d', fontsize = 14)
                elif type_copula.lower() == "cdf":
                    cbar.set_label("Cumulative probability", rotation=270, labelpad = 20,
                                    color = '#4d4d4d', fontsize = 14)
            else:
                raise ValueError("Plot method should be either '2d' or '3d'")
        
            
            if save_fig:
                print(save_path)
                plt.savefig(save_path, dpi=200)   


# def corr_diag(data, method='square', enrc = False, fontsize='medium', order=False,
#               title='Correlation diagram', own_data = None,
#               save_fig=False,
#               save_path= '../figures/correlation_diagram.png'):
#     """Plot a correlation diagram:
#         All parameters are plotted on the x-axis and the y-axis
        
        
#     Parameters
#     ----------
#     data : pandas.DataFrame
#         Dataframe with the variables
        
#     enrc : bool (optional)
#         Calculate empirical normal rank correlation
    
#     method : str (optional)
#         Default value: 'square'
#         The way it is plotted. Just try them to see what they do.   
#         Possibilities:
#         'ellipse', 'square', 'rectangle', 
#         'color', 'circle', 'number', 'text', 'pie'
        
#     fontsize : str (optional)
#         The fontsize of the text around it. Either, 'small',
#         'medium' or 'large'. Default value: 'medium'. 
        
#     order : bool (optional)
#         Here, the code behind it makes a nice order to group variables
#         with a high correlation. Default value = False because it 
#         is not always practical. 
#     """
#     cp = corrplot.Corrplot(data)
#     if own_data is not None:
#         cp.df = own_data
#     elif enrc:
#         enrcmatrix = pd.DataFrame(columns = data.columns, index = data.columns)
#         for var1 in data.columns:
#             for var2 in data.columns:
#                 c2d = Copula2d(data[var1].values, data[var2].values)
#                 enrc = c2d.emp_norm_rank_corr()
#                 enrcmatrix[var1][var2] = enrc
#         cp.df = enrcmatrix
#     cp.plot(method=method, fontsize=fontsize, order=order)
#     plt.title(title, fontsize = 20)
#     if save_fig:
#         plt.savefig(save_path, dpi=200, bbox_inches = "tight")
        

# def scatter_hist(data,
#                  name_x = 'x',
#                  name_y = 'y',
#         kargs_scatter={'s':10, 'c':'b'},
#         kargs_grids={},
#         kargs_histx={},
#         kargs_histy={},
#         scatter_position='bottom left',
#         width=.5,
#         height=.5,
#         offset_x=.10,
#         offset_y=.10,
#         gap=0.06,
#         facecolor='lightgrey',
#         grid=True,
#         show_labels=True,
#         **kargs):
#     """ Make a scatterplot of two parameters with a histogram of the margins
    
#     Parameters
#     ----------
#     data : numpy.array or pandas.DataFrame
#         The data
#     """
#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame({name_x: data[0], name_y: data[1]})
#     sh = ScatterHist(data)
#     plt.figure(figsize = (12,10))
#     sh.plot(kargs_scatter=kargs_scatter,
#         kargs_grids=kargs_grids,
#         kargs_histx=kargs_histx,
#         kargs_histy=kargs_histy,
#         scatter_position=scatter_position,
#         width=width,
#         height=height,
#         offset_x=offset_x,
#         offset_y=offset_y,
#         gap=gap,
#         facecolor=facecolor,
#         grid=grid,
#         show_labels=show_labels,
#         **kargs)
    

def pred_target(pred, target, 
                confidence_interval = None,
                dates=None, 
                data_name = 'Water level (m)',
                xlabel='Date', 
                title = 'Prediction and actual water level',                
                save_fig=False, 
                save_path='pred_target.png',
                conflabel = 'Confidence interval'
                ):
    """ Plot the prediction and target 
    
    Parameters
    ----------
    pred : numpy.array, list or pandas.Series
        The predicted values
        
    target : numpy.array, list or pandas.Series
        The observations    
        
    confidence_interval : numpy.array (optional)
        The confidence interval for this item, as timestep:lower,upper
    
    dates : list or pandas.Series of datetime.datetime objects (optional)
        xticks, when given. Otherwise xticks are all the whole numbers
    
    data_name : str (optional)
        Label on the y-axis
        
    xlabel : str (optional)
        Label on the x-axis
    
    title : str (optional)
        Title of the figure
    
    save_fig : bool (optional)
        Whether or not to save the figure.
        
    save_path : str (optional)
        Path where to save the figure to
        
    conflabel : str (optional)
        label of the confidence index
    """
    plt.figure(figsize=(11,7))
    if dates is not None:
        plt.plot(dates, target, linewidth = 1, linestyle=':',
                 marker='o', fillstyle='none', 
                 color = 'black', label = 'Observations', zorder=2)   
        plt.plot(dates, pred, linewidth = 2, label = 'Prediction', zorder=1)
        
        if confidence_interval is not None:
            plt.fill_between(dates, 
                     confidence_interval[:,0], confidence_interval[:,1],color='lightgrey',
                     zorder=0,
                         label=conflabel)
    else:
        plt.plot(target, linewidth = 1, linestyle=':',
                 marker='o', fillstyle='none', 
                 color = 'black', label = 'Observations', zorder=2)   
        plt.plot(pred, linewidth = 2, label = 'Prediction', zorder=1)
        if confidence_interval is not None:
            plt.fill_between(np.arange(len(pred)), 
                         confidence_interval[:,0], confidence_interval[:,1],color='lightgrey',
                         zorder=0,
                         label=conflabel)
    plt.grid(which='major', axis='y', color='darkgrey',
             linestyle='dashdot', linewidth=0.3)
    plt.legend()
    def simpleaxis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    ax = plt.subplot(111)
    simpleaxis(ax)
    plt.xlabel(xlabel,  color = '#4d4d4d', fontsize = 15)
    plt.ylabel(data_name,  color = '#4d4d4d', fontsize = 15)
    plt.xticks( color = '#4d4d4d', fontsize = 15)
    plt.yticks( color = '#4d4d4d', fontsize = 15)
    plt.title(title,  fontsize = 18)
    #plt.axhline(0, color = 'grey')
    #plt.ylim(0,15)
    
    
    
    if save_fig:
        plt.savefig(save_path, dpi=200)
        
        
def cumulative_plot(data_dict, 
                    start_date = None,
                    end_date = None,
                    ylabel = "Cumulative precipitation (mm)",
                    title = "Comparision cumulative precipitation",
                    save_fig = False,
                    save_path = "../../figures/cumulative.png"):
    
    """Make a cumulative plot of for example the precipitation.
    
    Parameters
    ----------
    data_dict : dictionary with pandas.Series
        The data. Key is name of the data
        
    start_date : datetime.DateTime (optional)
        The start date.
        Takes all to end date when not specified.
        
    end_date : datetime.DateTime (optional)
        The end date.    
        Takes all data from start date when not specified.
    
    ylabel : str (optional)
        The ylabel of the plot.
        Default value: "Cumulative precipitation (mm)"
        
    title : str (optional)
        The title of the plot. 
        Default value: "Comparision cumulative precipitation"
        
    save_fig : bool (optional)
        Whether or not to save the figure.
        
    save_path : str (optional)
        Path where to save the figure to
    
    """
    
    plt.figure(figsize=(12,8))
    
    for src in data_dict:
        if start_date is not None:
            data = data_dict[src][data_dict[src].index>=start_date].copy()
        else:
            data = data_dict[src].copy()
        if end_date is not None:
            data = data[data.index<end_date]
            
        data = data.fillna(0)
        for i in range(len(data)-1):
            data[i+1] = data[i+1]+data[i]
        
        plt.plot(data, label = src)
        
        
        print(src)
        print(data[np.logical_and(data.index > dt.datetime(2018,6,1), data.index < dt.datetime(2018,6,2))])
        
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)        
    plt.legend()
    if save_fig:
        plt.savefig(save_path, dpi=200)

    
def double_hist(data_uncond, data_cond,
                data_name = "Discharge $(m^3/s)$",
                title = "Histogram conditioned and unconditioned variables",
                bins = 16,
                save_fig = False,
                save_path = "../../figures/double_hist.png"):
    """Make a histogram of data and conditionalized expectation.
    
    Parameters
    ----------
    data_uncond : numpy.array, list or pandas.Series
        Unconditional data
        
    data_cond : numpy.array, list or pandas.Series
        Conditional data
        
    title : str (optional)
        Title of the figure
        
    bins : int or sequence or str (optional)
        Default value: 16
        If an integer is given, bins + 1 bin 
        edges are calculated and returned, consistent with numpy.histogram.
        If bins is a sequence, gives bin edges, 
        including left edge of first bin and right edge of last bin. 
        In this case, bins is returned unmodified.
        All but the last (righthand-most) bin is 
        half-open. In other words, if bins is:  
        then the first bin is [1, 2) (including 1, but excluding 2) 
        and the second [2, 3). The last bin, however, is [3, 4], which
        includes 4.
        Unequally spaced bins are supported if bins is a sequence.
        With Numpy 1.11 or newer, you can alternatively provide a 
        string describing a binning strategy, such as 'auto', 
        'sturges', 'fd', 'doane', 'scott', 'rice' or 'sqrt', 
        see numpy.histogram.
                   
    save_fig : bool (optional)
        Whether or not to save the figure.
        
    save_path : str (optional)
        Path where to save the figure to
    
    """
    

    plt.figure(figsize=(12,8))
    _,bins2,_ = plt.hist(data_uncond,  density=True,zorder=2, color=(0.6,0.6,0.6), 
                        label="Unconditioned", bins = bins) #histtype=u'step',
    counts,_,_ = plt.hist(data_cond,density=True,
                          zorder=3,alpha=0.6, color='navy', 
                          label="Conditioned", bins=bins2, 
                          edgecolor='darkblue',
                          rwidth=0.8)
    if not np.max(counts) > 0.35/(bins2[1]-bins2[0]):
        plt.ylim(0,0.35/(bins2[1]-bins2[0]))
    plt.legend()
    
    
    plt.grid(which='major', axis='y', color='darkgrey',
                 linestyle='dashdot', linewidth=0.3)
    
    plt.xlabel(data_name)
    plt.ylabel("Change of occurence")
    plt.title(title)
    
    if save_fig:
        plt.savefig(save_path, dpi=200)
    