# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 23:18:07 2022

@author: Dr.Yu
"""
from scipy import ndimage
import numpy as np
import copy
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
from matplotlib.ticker import  MultipleLocator


memname=['imerg','cmorph','gsmap','era5','wrf','wrfn']

fig,axs=plt.subplots(nrows=4, ncols=3, figsize=(14,16))
fig.subplots_adjust(wspace=0.35,hspace=0.35)


dataname=['IMERG','CMORPH','GSMaP','ERA5','WRF','WRF$_{nudging}$']

for i in range(len(memname)):
    locals()[memname[i]]=joblib.load('./supercompkl/'+memname[i]+'_shapemat_BTmax.pkl')
    
lon=np.linspace(-3,3,61)
lat=np.linspace(-3,3,61)
y,x=np.meshgrid(lat,lon)

extent=[-3,3,-3,3]

for i in range(len(memname)):
    
    
    thisvar=np.nanmean(locals()[memname[i]][:,:,:],axis=0)    
    ax=plt.subplot(4,3,i+1)
    thispic1=ax.pcolormesh(x,y,thisvar,cmap=cmaps.CBR_coldhot,alpha=0.75,vmin=219,vmax=241)#######
    print(np.nanmin(thisvar))
    
    ax.set_xticks([-3,-1.5,0,1.5,3])
    ax.set_yticks([-3,-1.5,0,1.5,3])
    ax.set_xticklabels([-3,-1.5,0,1.5,3],fontsize=13)
    ax.set_yticklabels([-3,-1.5,0,1.5,3],fontsize=13)
    xminorLocator = MultipleLocator(0.2)
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = MultipleLocator(0.2)
    ax.yaxis.set_minor_locator(yminorLocator)
    
    if i in [3,4,5]:
        ax.set_xlabel('Distance from the centre (°)',fontsize=14)
    if i in [0,3]:
        ax.set_ylabel('Distance from the centre (°)',fontsize=14)

    ax.set_title(dataname[i]+' Minimum: '+str(round(np.nanmin(thisvar),2))+'K',loc='left', pad=5,fontsize=14)
    ax.axhline(y=0,ls='--',c='grey',lw=0.6)
    ax.axvline(x=0,ls='--',c='grey',lw=0.6)

ax.text(-20.5,11.65,'(a)',fontsize=20)

l = 0.95
b = 0.55
w = 0.015
h = 0.3
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic1, cax=cbar_ax,extend='max',orientation='vertical')
cb.set_label('Brightness Temperature (K)',fontsize=15)
cb.set_ticks(np.linspace(219,241,12))
cb.ax.tick_params(labelsize=13)

##############################################################################
##############################################################################

for i in range(len(memname)):
    locals()[memname[i]]=joblib.load('./supercompkl/'+memname[i]+'_shapemat_BTmaxPLUS.pkl')


for i in range(len(memname)):    
    
    thisvar=np.nanmean(locals()[memname[i]][:,:,:],axis=0)    
    ax=plt.subplot(4,3,i+7)
    thispic1=ax.pcolormesh(x,y,thisvar,cmap=cmaps.CBR_coldhot,alpha=0.75,vmin=219,vmax=241)#######
    print(np.nanmin(thisvar))
    
    ax.set_xticks([-3,-1.5,0,1.5,3])
    ax.set_yticks([-3,-1.5,0,1.5,3])
    ax.set_xticklabels([-3,-1.5,0,1.5,3],fontsize=13)
    ax.set_yticklabels([-3,-1.5,0,1.5,3],fontsize=13)
    xminorLocator = MultipleLocator(0.2)
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = MultipleLocator(0.2)
    ax.yaxis.set_minor_locator(yminorLocator)
    
    if i in [3,4,5]:
        ax.set_xlabel('Distance from the centre (°)',fontsize=14)
    if i in [0,3]:
        ax.set_ylabel('Distance from the centre (°)',fontsize=14)

    ax.set_title(dataname[i]+' Minimum: '+str(round(np.nanmin(thisvar),2))+'K',loc='left', pad=5,fontsize=13)
    ax.axhline(y=0,ls='--',c='grey',lw=0.6)
    ax.axvline(x=0,ls='--',c='grey',lw=0.6)

ax.text(-20.5,11.65,'(b)',fontsize=20)

l = 0.95
b = 0.14
w = 0.015
h = 0.3
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic1, cax=cbar_ax,extend='max',orientation='vertical')
cb.set_label('Brightness Temperature (K)',fontsize=15)
cb.set_ticks(np.linspace(219,241,12))
cb.ax.tick_params(labelsize=13)


fig.savefig('NormalFigureSP2.BTstructure.jpg',bbox_inches='tight',dpi=300)