# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:04:45 2024

@author: Dr.Yu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.colors as mc
import xarray as xr
import cmaps
import copy
from matplotlib.ticker import  MultipleLocator
import matplotlib
from matplotlib.patches import ConnectionPatch


fig,axs=plt.subplots(nrows=3, ncols=2, figsize=(14,9))
fig.subplots_adjust(wspace=0.25,hspace=0.45)

widthlist=[-0.25,-0.15,-0.05,0.05,0.15,0.25]
tracklist=['imerg','cmorph','gsmap','era5','wrf','wrfn']
labels=['IMERG','CMORPH','GSMaP','ERA5','WRF','WRF$_{nudging}$']
colors=['red','hotpink','firebrick','orange','cornflowerblue','green']
titles=['(a)Number','(b)Lifetime','(c)Area','(d)MaxPrecip','(e)MeanPrecip']
ylabels=['Number of Tracked MCSs','(hr)','(km$^{2}$)','(mm/hr)','(mm/hr)']

ax1 = plt.subplot(3,1,1)
for i in range(len(tracklist)):
    locals()[tracklist[i]]=joblib.load('./supercompkl/'+tracklist[i]+'_statn.pkl')
    #ax1.plot(range(12),locals()[tracklist[i]][:,0],label=labels[i],color=colors[i])
    ax1.bar(np.arange(0,12)+widthlist[i],locals()[tracklist[i]][:,0],width=0.1,label=labels[i],color=colors[i])
    print(tracklist[i],np.sum(locals()[tracklist[i]][7:,0]))
# ax1.axvline(x=7,color='k',ls='--')
# ax1.axvline(x=11,color='k',ls='--')
ax1.axhline(y=10,color='grey',ls='--',lw=1)
ax1.set_xticks(range(12))
ax1.set_xticklabels(['O','N','D','J','F','M','A','M','J','J','A','S'],fontsize=12)
ax1.set_xlabel('Month',fontsize=14)
ax1.set_ylabel(ylabels[0],fontsize=14)
ax1.set_title(titles[0],loc='left', pad=5,fontsize=16)
ax1.legend(ncol=6,loc=4,bbox_to_anchor=(0.965, -3.55),fontsize=14)


widthlist=[-0.15,-0.05,0.05,0.15]
tracklist=['imerg','cmorph','wrf','wrfn']
labels=['IMERG','CMORPH','WRF','WRF_nudging']
colors=['red','hotpink','cornflowerblue','green']

for ploti in range(4):
    ax1 = plt.subplot(3,2,ploti+3)    
    for i in range(len(tracklist)):
        locals()[tracklist[i]]=joblib.load('./supercompkl/'+tracklist[i]+'_statn.pkl')
        if ploti==2:
            thevar=locals()[tracklist[i]][7:,4]
        elif ploti==3:
            thevar=locals()[tracklist[i]][7:,3]
        else:
            thevar=locals()[tracklist[i]][7:,ploti+1]
        thevar[thevar==9999]=np.nan
        if ploti==1:
            thevar=thevar/1000000
        #ax1.plot(range(5),thevar,label=labels[i],color=colors[i])
        ax1.bar(np.arange(0,5)+widthlist[i],thevar,width=0.1,label=labels[i],color=colors[i])

    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['M','J','J','A','S'],fontsize=12)
    ax1.set_xlabel('Month',fontsize=14)
    ax1.set_ylabel(ylabels[ploti+1],fontsize=14)
    ax1.set_title(titles[ploti+1],loc='left', pad=5,fontsize=16)
    if ploti==3:
        ax1.set_yticklabels(['{:.1f}'.format(x) for x in ax1.get_yticks()],fontsize=12)
    else:
        ax1.set_yticklabels(['{:.0f}'.format(x) for x in ax1.get_yticks()],fontsize=12)
        
fig.savefig('NormalFigure1.STAT.jpg',bbox_inches='tight',dpi=300)