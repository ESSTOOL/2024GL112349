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



'''
daf=joblib.load('./imerg_202008.nc_grMCSs.pkl')
ddf=joblib.load('./imerg_202008.mcsinfo.pkl')
'''

plotha1=[1,2,5,6,9,10]
plotha2=[13,14,17,18,21,22]
plotha3=[3,4,7,8,11,12]
plotha4=[15,16,19,20,23,24]

memname=['imerg','cmorph','gsmap','era5','wrf','wrfn']

fig,axs=plt.subplots(nrows=6, ncols=4, figsize=(19,24))
fig.subplots_adjust(wspace=0.3,hspace=0.35)


dataname=['IMERG','CMORPH','GSMaP','ERA5','WRF','WRF$_{nudging}$']

for i in range(len(memname)):
    locals()[memname[i]]=joblib.load('./supercompkl/'+memname[i]+'_shapemat_PRmax.pkl')
    
lon=np.linspace(-1,1,21)
lat=np.linspace(-1,1,21)
y,x=np.meshgrid(lat,lon)

extent=[-1,1,-1,1]

for i in range(len(memname)):
    
    
    thisvar=np.nanmean(locals()[memname[i]][:,20:41,20:41],axis=0)    
    ax=plt.subplot(6,4,plotha1[i])
    thispic1=ax.pcolormesh(x,y,thisvar,cmap=cmaps.perc2_9lev,alpha=0.75,vmin=0,vmax=40)#######
    print(np.nanmax(thisvar))
    
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=13)
    ax.set_yticklabels([-1,-0.5,0,0.5,1],fontsize=13)
    xminorLocator = MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = MultipleLocator(0.1)
    ax.yaxis.set_minor_locator(yminorLocator)
    
    if i in [4,5]:
        ax.set_xlabel('Distance from the centre (°)',fontsize=14)
    if i in [0,2,4]:
        ax.set_ylabel('Distance from the centre (°)',fontsize=14)

    ax.set_title(dataname[i]+': '+str(round(np.nanmax(thisvar),2))+' mm/hr',loc='left', pad=5,fontsize=15)
    ax.axhline(y=0,ls='--',c='grey',lw=0.6)
    ax.axvline(x=0,ls='--',c='grey',lw=0.6)

    if i==0:
        ax.text(-1.4,1.3,'(a)',fontsize=22)

l = 0.03
b = 0.525
w = 0.015
h = 0.35
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic1, cax=cbar_ax,extend='max',orientation='vertical')
cb.set_label('Precipitation (mm/hr)',fontsize=15)
cb.set_ticks(np.linspace(0,40,11))
cb.ax.tick_params(labelsize=13)

##############################################################################上为子图1
##############################################################################下为子图2

for i in range(len(memname)):
    locals()[memname[i]]=joblib.load('./supercompkl/'+memname[i]+'_shapemat_PRmaxPLUS.pkl')


for i in range(len(memname)):    
    
    thisvar=np.nanmean(locals()[memname[i]][:,20:41,20:41],axis=0)    
    ax=plt.subplot(6,4,plotha2[i])
    thispic1=ax.pcolormesh(x,y,thisvar,cmap=cmaps.perc2_9lev,alpha=0.75,vmin=0,vmax=20)#######
    print(np.nanmax(thisvar))
    
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=13)
    ax.set_yticklabels([-1,-0.5,0,0.5,1],fontsize=13)
    xminorLocator = MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = MultipleLocator(0.1)
    ax.yaxis.set_minor_locator(yminorLocator)
    
    if i in [4,5]:
        ax.set_xlabel('Distance from the centre (°)',fontsize=14)
    if i in [0,2,4]:
        ax.set_ylabel('Distance from the centre (°)',fontsize=14)

    ax.set_title(dataname[i]+': '+str(round(np.nanmax(thisvar),2))+' mm/hr',loc='left', pad=5,fontsize=15)
    ax.axhline(y=0,ls='--',c='grey',lw=0.6)
    ax.axvline(x=0,ls='--',c='grey',lw=0.6)

    if i==0:
        ax.text(-1.4,1.3,'(c)',fontsize=22)

l = 0.03
b = 0.13
w = 0.015
h = 0.35
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic1, cax=cbar_ax,extend='max',orientation='vertical')
cb.set_label('Precipitation (mm/hr)',fontsize=15)
cb.set_ticks(np.linspace(0,20,11))
cb.ax.tick_params(labelsize=13)


##############################################################################上为子图2
##############################################################################下为子图3


import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import cmaps
# z_list=np.array([50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,
#                  650,700,750,775,800,825,850,875,900,925,950,975,1000])

z_list=np.log10(np.array([200,225,250,300,350,400,450,500,550,600,
                  650,700,750,775,800,825,850,875,900,925,950,975,1000]))

memname=['imerg','cmorph','gsmap','era5','wrf','wrfn']
dataname=['IMERG+ERA5: ','CMORPH+ERA5: ','GSMaP+ERA5: ','ERA5: ','WRF: ','WRF$_{nudging}$: ']

def xinjiafa(x,y):
    newmat=np.full((x.shape[0],x.shape[1],x.shape[2]),np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                xijk=x[i,j,k]
                yijk=y[i,j,k]
                if ((np.isnan(xijk)==True) and (np.isnan(yijk)==True)):
                    newmat[i,j,k]=np.nan
                elif ((np.isnan(xijk)==False) and (np.isnan(yijk)==False)):
                    newmat[i,j,k]=(xijk+yijk)/2
                elif ((np.isnan(xijk)==True) and (np.isnan(yijk)==False)):
                    newmat[i,j,k]=yijk
                elif ((np.isnan(xijk)==False) and (np.isnan(yijk)==True)):
                    newmat[i,j,k]=xijk
    return(newmat)

for i in range(6):#6
    axs=plt.subplot(6,4,plotha3[i])
    data=joblib.load('./'+memname[i]+'_cube.pkl')
    data[data>1000]=np.nan
    if i<=3:    
        xera5_list=np.array(range(-11,12))
        x,y=np.meshgrid(xera5_list,z_list)
        
        
        theept=np.nanmean(xinjiafa(data[0,:,6:,11,:],np.flip(data[0,:,6:,:,11],axis=2)),axis=0)
        #theept=xinjiafa(data[0,:,6:,11,:],np.flip(data[0,:,6:,:,11],axis=2))[4,:,:]
       
        theomg=np.nanmean(xinjiafa(data[1,:,6:,11,:],np.flip(data[1,:,6:,:,11],axis=2)),axis=0)
        #theomg=xinjiafa(data[1,:,6:,11,:],np.flip(data[1,:,6:,:,11],axis=2))[4,:,:]
        
        thispic1=axs.pcolormesh(x,y,theept,cmap=cmaps.CBR_coldhot[[0,1,2,3,4,6,7,8,9,10]],vmin=313,vmax=353)
        if i<=1:
            thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.15,-0.3,-0.45,-0.6,-0.75]),colors='k',alpha=0.75)
        elif i==2:
            thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.3,-0.6,-0.9,-1.2,-1.5]),colors='k',alpha=0.75)
        elif i==3:
            thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.5,-1.25,-2,-2.75]),colors='k',alpha=0.75)
        #print(np.nanmin(omgera5))
        axs.clabel(thiscon1, inline=True, fontsize=9)
        axs.set_yticks(np.log10([200,300,500,700,850,1000]))
        axs.set_yticklabels([200,300,500,700,850,1000],fontsize=11)
        axs.invert_yaxis()
        axs.set_xticks([-11,-5.5,0,5.5,11])
        axs.set_xticklabels([-260,-130,0,130,260],fontsize=11)
        if i==0:        
            axs.text(-15,2.23,'(b)',fontsize=22)
    else:
        xwrf_list=np.array(range(-73,74))
        x,y=np.meshgrid(xwrf_list,z_list)
        theept=np.nanmean(xinjiafa(data[0,:,6:,73,:],data[0,:,6:,:,73]),axis=0)
        #eptwrf[eptwrf>=400]=np.nan
        #omgwrf=np.nanmean((data[1,:,6:,73,:]+data[1,:,6:,:,73])/2,axis=0)
        theomg=np.nanmean(xinjiafa(data[1,:,6:,73,:],data[1,:,6:,:,73]),axis=0)
        #omgwrf[omgwrf>=50]=np.nan
        thispic1=axs.pcolormesh(x,y,theept,cmap=cmaps.CBR_coldhot[[0,1,2,3,4,6,7,8,9,10]],vmin=313,vmax=353)
        thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.6,-1.8,-3]),colors='k',alpha=0.75)
        #print(np.nanmin(omgwrf))
        axs.clabel(thiscon1, inline=True, fontsize=9)
        axs.set_yticks(np.log10([200,300,500,700,850,1000]))
        axs.set_yticklabels([200,300,500,700,850,1000],fontsize=11)
        axs.invert_yaxis()
        axs.set_xticks([-73,-36.5,0,36.5,73])
        axs.set_xticklabels([-260,-130,0,130,260],fontsize=11)        

    if i in [0,2,4]:
        axs.set_ylabel('Pressure (hPa)',labelpad=0,fontsize=14)

    if i in [4,5]:
        axs.set_xlabel('Distance from the centre (km)',fontsize=14)

    axs.set_title(dataname[i]+str(round(np.nanmin(theomg),2))+' Pa/s',loc='left', pad=5,fontsize=15)

l = 0.93
b = 0.525
w = 0.015
h = 0.35
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic1, cax=cbar_ax,extend='both',orientation='vertical')
cb.set_label('Equivalent Potential Temperature (K)',fontsize=15)
cb.set_ticks(np.linspace(313,353,11))
cb.ax.tick_params(labelsize=13)

##############################################################################上为子图3
##############################################################################下为子图4

for i in range(6):#6
    axs=plt.subplot(6,4,plotha4[i])
    data=joblib.load('./'+memname[i]+'_cube2.pkl')
    data[data>1000]=np.nan
    if i<=3:    
        xera5_list=np.array(range(-11,12))
        x,y=np.meshgrid(xera5_list,z_list)
        
        
        theept=np.nanmean(xinjiafa(data[0,:,6:,11,:],np.flip(data[0,:,6:,:,11],axis=2)),axis=0)
        #theept=xinjiafa(data[0,:,6:,11,:],np.flip(data[0,:,6:,:,11],axis=2))[4,:,:]
       
        theomg=np.nanmean(xinjiafa(data[1,:,6:,11,:],np.flip(data[1,:,6:,:,11],axis=2)),axis=0)
        #theomg=xinjiafa(data[1,:,6:,11,:],np.flip(data[1,:,6:,:,11],axis=2))[4,:,:]
        
        thispic1=axs.pcolormesh(x,y,theept,cmap=cmaps.CBR_coldhot[[0,1,2,3,4,6,7,8,9,10]],vmin=313,vmax=353)
        if i<=1:
            thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.15,-0.3,-0.45,-0.6,-0.75]),colors='k',alpha=0.75)
        elif i==2:
            thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.3,-0.6,-0.9,-1.2,-1.5]),colors='k',alpha=0.75)
        elif i==3:
            thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.5,-1.25,-2,-2.75]),colors='k',alpha=0.75)
        #print(np.nanmin(omgera5))
        axs.clabel(thiscon1, inline=True, fontsize=9)
        axs.set_yticks(np.log10([200,300,500,700,850,1000]))
        axs.set_yticklabels([200,300,500,700,850,1000],fontsize=11)
        axs.invert_yaxis()
        axs.set_xticks([-11,-5.5,0,5.5,11])
        axs.set_xticklabels([-260,-130,0,130,260],fontsize=11)
        if i==0:        
            axs.text(-15,2.23,'(d)',fontsize=22)
    else:
        xwrf_list=np.array(range(-73,74))
        x,y=np.meshgrid(xwrf_list,z_list)
        theept=np.nanmean(xinjiafa(data[0,:,6:,73,:],data[0,:,6:,:,73]),axis=0)
        #eptwrf[eptwrf>=400]=np.nan
        #omgwrf=np.nanmean((data[1,:,6:,73,:]+data[1,:,6:,:,73])/2,axis=0)
        theomg=np.nanmean(xinjiafa(data[1,:,6:,73,:],data[1,:,6:,:,73]),axis=0)
        #omgwrf[omgwrf>=50]=np.nan
        thispic1=axs.pcolormesh(x,y,theept,cmap=cmaps.CBR_coldhot[[0,1,2,3,4,6,7,8,9,10]],vmin=313,vmax=353)
        thiscon1=axs.contour(x,y,theomg,levels=np.flip([-0.6,-1.8,-3]),colors='k',alpha=0.75)
        #print(np.nanmin(omgwrf))
        axs.clabel(thiscon1, inline=True, fontsize=9)
        axs.set_yticks(np.log10([200,300,500,700,850,1000]))
        axs.set_yticklabels([200,300,500,700,850,1000],fontsize=11)
        axs.invert_yaxis()
        axs.set_xticks([-73,-36.5,0,36.5,73])
        axs.set_xticklabels([-260,-130,0,130,260],fontsize=11)        

    if i in [0,2,4]:
        axs.set_ylabel('Pressure (hPa)',labelpad=0,fontsize=14)

    if i in [4,5]:
        axs.set_xlabel('Distance from the centre (km)',fontsize=14)

    axs.set_title(dataname[i]+str(round(np.nanmin(theomg),2))+' Pa/s',loc='left', pad=5,fontsize=15)

l = 0.93
b = 0.13
w = 0.015
h = 0.35
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic1, cax=cbar_ax,extend='both',orientation='vertical')
cb.set_label('Equivalent Potential Temperature (K)',fontsize=15)
cb.set_ticks(np.linspace(313,353,11))
cb.ax.tick_params(labelsize=13)


fig.savefig('newNormalFigure3.PRstructure.jpg',bbox_inches='tight',dpi=300)