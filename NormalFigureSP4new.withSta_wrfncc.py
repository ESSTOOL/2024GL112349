# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:15:58 2024

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

def deletehang(mat,thres):
    ###
    # mat = np.array([[1, 2, np.nan, 4],
    #                 [5, np.nan, 7, 8],
    #                 [9, 10, 11, np.nan],
    #                 [13, 14, 15, 16],
    #                 [17, 18, 19, 20],
    #                 [np.nan, np.nan, np.nan, np.nan]])
    # 计算每一行中nan值的数量
    nan_counts = np.isnan(mat).sum(axis=1)  
    # 判断哪些行需要删除
    threshold = thres  # 设定阈值为30%就输入0.3
    delete_mask = nan_counts > threshold * mat.shape[1]   
    # 删除需要删除的行
    mat = mat[~delete_mask]
    return(mat,~delete_mask)
###

def getline(data):
    data=data[~np.isnan(data)]
    data[data<0.1]=0
    totalnum=data.shape[0]
    distrix=np.linspace(0,100,101)
    frequency=np.zeros(distrix.shape[0])
    for i in range(len(distrix)):
        if i==0:
            frequency[0]=np.sum(data<=0)/totalnum
        else:
            frequency[i]=np.sum((data>distrix[i-1])&(data<=distrix[i]))/totalnum
    return(frequency)

def moving_average(x):#只能用于5个的！！！！！
    x=np.array(x)
    mat=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if i==0:
            mat[i]=(x[0]+x[1]+x[2])/3
        elif i==1:
            mat[i]=(x[0]+x[1]+x[2]+x[3])/4
        elif i==(x.shape[0]-2):
            mat[i]=(x[i-2]+x[i-1]+x[i]+x[i+1])/4
        elif i==(x.shape[0]-1):
            mat[i]=(x[i-2]+x[i-1]+x[i])/3
        else:
            mat[i]=(x[i-2]+x[i-1]+x[i]+x[i+1]+x[i+2])/5
    return(mat)            

tracklist=['imerg','cmorph','gsmap','era5','wrfn']
labels=['IMERG','CMORPH','GSMaP','ERA5','WRF$_{nudging}$']

fig,axs=plt.subplots(nrows=4, ncols=2, figsize=(13,18))
fig.subplots_adjust(wspace=0.25,hspace=0.45)

varname=['imerg','cmorph','gsmap','era5','wrfn']
color=['red','hotpink','firebrick','orange','green']
titlename=['IMERG','CMORPH','GSMaP','ERA5','WRF_nudging']

obs=np.loadtxt('stations1yr.txt')[:,5112:]
obs[obs>300]=np.nan

ax1 = plt.subplot(421)
ax1.set_yscale('log')
ax1.set_xscale('log')
#sub_axes.set_xscale('log')
for i in range(5):
    data=np.loadtxt(tracklist[i]+'1yr_mcs.nc.mcsn.txt')[:,5112:]
    datamcs=np.loadtxt('wrfn1yr.txt')[:,5112:]
    data[datamcs==0]=np.nan
    data[np.isnan(obs)]=np.nan
    thisvar0=data.flatten()
    thisvar1=thisvar0[~np.isnan(thisvar0)]
    thisvar2=getline(thisvar1)
    thisvar3=moving_average(thisvar2)
    ax1.plot(np.linspace(0,100,101),thisvar3,color=color[i],linewidth=1.5,label=titlename[i],alpha=1)

dataobs=copy.deepcopy(obs)
dataobs[datamcs==0]=np.nan
thisvar0=dataobs.flatten()
thisvar1=thisvar0[~np.isnan(thisvar0)]
thisvar2=getline(thisvar1)
thisvar3=moving_average(thisvar2)
ax1.plot(np.linspace(0,100,101),thisvar3,color='k',linewidth=1.5,label='OBS',alpha=1)


ax1.set_xlabel('Precipitation (mm/hr)',fontsize=15)
ax1.set_ylabel('Probability',fontsize=15)
xminorLocator = MultipleLocator(5)
ax1.xaxis.set_minor_locator(xminorLocator)
ax1.set_xlim([1,100])
#ax1.set_ylim([0.000003,0.5])
ax1.set_xticks([1,2,3,4,5,10,15,20,30,40,50,100])
ax1.set_xticklabels(['1','','','','5','10','','20','30','40','50','100'],fontsize=10)
ax1.set_title('(a)MCS-Associated Precipitation',loc='left', pad=5,fontsize=15)


ax1 = plt.subplot(422)
ax1.set_yscale('log')
ax1.set_xscale('log')
#sub_axes.set_xscale('log')
for i in range(5):
    data=np.loadtxt(tracklist[i]+'1yr_mcs.nc.mcsn.txt')[:,5112:]
    datamcs=np.loadtxt('wrfn1yr.txt')[:,5112:]
    data[datamcs>0]=np.nan
    data[np.isnan(obs)]=np.nan
    thisvar0=data.flatten()
    thisvar1=thisvar0[~np.isnan(thisvar0)]
    thisvar2=getline(thisvar1)
    thisvar3=moving_average(thisvar2)
    ax1.plot(np.linspace(0,100,101),thisvar3,color=color[i],linewidth=1.5,label=titlename[i],alpha=1)

dataobs=copy.deepcopy(obs)
dataobs[datamcs>0]=np.nan
thisvar0=dataobs.flatten()
thisvar1=thisvar0[~np.isnan(thisvar0)]
thisvar2=getline(thisvar1)
thisvar3=moving_average(thisvar2)
ax1.plot(np.linspace(0,100,101),thisvar3,color='k',linewidth=1.5,label='OBS',alpha=1)

ax1.set_xlabel('Precipitation (mm/hr)',fontsize=15)
ax1.set_ylabel('Probability',fontsize=15)
xminorLocator = MultipleLocator(5)
ax1.xaxis.set_minor_locator(xminorLocator)
ax1.set_xlim([1,100])
#ax1.set_ylim([0.000003,0.5])
ax1.set_xticks([1,2,3,4,5,10,15,20,30,40,50,100])
ax1.set_xticklabels(['1','','','','5','10','','20','30','40','50','100'],fontsize=10)
ax1.set_title('(b)nonMCS-Associated Precipitation',loc='left', pad=5,fontsize=15)

ax1.legend(ncol=6,loc=4,bbox_to_anchor=(0.85, 1.2),fontsize=12)


########################################################日变化

tracklist=['stations','imerg','cmorph','gsmap','era5','wrfn']
labels=['OBS','IMERG','CMORPH','GSMaP','ERA5','WRF$_{nudging}$']

color=['k','red','hotpink','firebrick','orange','green']

obs=np.loadtxt('stations1yr.nc.txt')[:,5112:8784]
obs[obs>300]=np.nan
for i in range(2):
    ax1 = plt.subplot(4,2,i+3)
    for j in range(6):
        data=np.loadtxt(tracklist[j]+'1yr_mcs.nc.mcsn.txt')[:,5112:8784]
        datamcs=np.loadtxt('wrfn1yr.txt')[:,5112:8784]
        if i==0:
            data[datamcs==0]=np.nan
        elif i==1:
            data[datamcs>0]=np.nan
        data[np.isnan(obs)]=np.nan
        
        thexulie=[]
        for k in range(24):
            thexulie.append(np.nanmean(np.nanmean(data[:,k::24],axis=1),axis=0))
        thexulie=thexulie[-9:]+thexulie[:-9]
        ax1.plot(range(24),thexulie,color=color[j],linewidth=1.5,label=labels[j],alpha=1)    

    
    xminorLocator = MultipleLocator(1)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.set_xlim([0,23])
    ax1.set_xticks([0,4,8,12,16,20,23])
    ax1.set_xticklabels(['0','4','8','12','16','20','BJT'],fontsize=13)
    ax2 = ax1.twiny()
    ax2.set_xticks([0,4,8,12,16,20,23])
    ax2.set_xlim([0,23])
    ax2.set_xticklabels(['16','20','0','4','8','12','UTC'],fontsize=13)
    ax2.xaxis.set_minor_locator(xminorLocator)
    ax1.set_ylabel('(mm/hr)',fontsize=15)
    ax1.set_xlabel('Time',fontsize=15)

    
    if i==0:
        ax1.set_title('(c)Precipitation Amount (MCS)',loc='left', pad=5,fontsize=16)
        ax1.set_yticklabels(['{:.1f}'.format(x) for x in ax1.get_yticks()],fontsize=13)
    elif i==1:
        ax1.set_title('(d)Precipitation Amount (nonMCS)',loc='left', pad=5,fontsize=16)    
        ax1.set_yticklabels(['{:.2f}'.format(x) for x in ax1.get_yticks()],fontsize=13)

    
for i in range(2):
    ax1 = plt.subplot(4,2,i+5)
    for j in range(6):
        data=np.loadtxt(tracklist[j]+'1yr_mcs.nc.mcsn.txt')[:,5112:8784]
        datamcs=np.loadtxt('wrfn1yr.txt')[:,5112:8784]
        if i==0:
            data[datamcs==0]=np.nan
        elif i==1:
            data[datamcs>0]=np.nan
        data[np.isnan(obs)]=np.nan
        
        data[data>=0.1]=1
        data[data<0.1]=0
        
        thexulie=[]
        for k in range(24):
            thexulie.append(np.nanmean(np.nanmean(data[:,k::24],axis=1),axis=0))
        thexulie=thexulie[-9:]+thexulie[:-9]
        ax1.plot(range(24),thexulie,color=color[j],linewidth=1.5,label=labels[j],alpha=1)    

    xminorLocator = MultipleLocator(1)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.set_xlim([0,23])
    ax1.set_xticks([0,4,8,12,16,20,23])
    ax1.set_xticklabels(['0','4','8','12','16','20','BJT'],fontsize=13)
    ax2 = ax1.twiny()
    ax2.set_xticks([0,4,8,12,16,20,23])
    ax2.set_xlim([0,23])
    ax2.set_xticklabels(['16','20','0','4','8','12','UTC'],fontsize=13)
    ax2.xaxis.set_minor_locator(xminorLocator)
    ax1.set_ylabel('Frequency',fontsize=15)
    ax1.set_xlabel('Time',fontsize=15)
    
    if i==0:
        ax1.set_title('(e)Precipitation Frequency (MCS)',loc='left', pad=5,fontsize=16)
        ax1.set_yticklabels(['{:.1f}'.format(x) for x in ax1.get_yticks()],fontsize=13)
    elif i==1:
        ax1.set_title('(f)Precipitation Frequency (nonMCS)',loc='left', pad=5,fontsize=16)    
        ax1.set_yticklabels(['{:.3f}'.format(x) for x in ax1.get_yticks()],fontsize=13)


for i in range(2):
    ax1 = plt.subplot(4,2,i+7)
    for j in range(6):
        data=np.loadtxt(tracklist[j]+'1yr_mcs.nc.mcsn.txt')[:,5112:8784]
        datamcs=np.loadtxt('wrfn1yr.txt')[:,5112:8784]
        if i==0:
            data[datamcs==0]=np.nan
        elif i==1:
            data[datamcs>0]=np.nan
        data[np.isnan(obs)]=np.nan
        
        data[data<0.1]=np.nan
        
        thexulie=[]
        for k in range(24):
            thexulie.append(np.nanmean(np.nanmean(data[:,k::24],axis=1),axis=0))
        thexulie=thexulie[-9:]+thexulie[:-9]
        ax1.plot(range(24),thexulie,color=color[j],linewidth=1.5,label=labels[j],alpha=1)    

    xminorLocator = MultipleLocator(1)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.set_xlim([0,23])
    ax1.set_xticks([0,4,8,12,16,20,23])
    ax1.set_xticklabels(['0','4','8','12','16','20','BJT'],fontsize=13)
    ax2 = ax1.twiny()
    ax2.set_xticks([0,4,8,12,16,20,23])
    ax2.set_xlim([0,23])
    ax2.set_xticklabels(['16','20','0','4','8','12','UTC'],fontsize=13)
    ax2.xaxis.set_minor_locator(xminorLocator)
    ax1.set_ylabel('(mm/hr)',fontsize=15)
    ax1.set_xlabel('Time',fontsize=15)
    
    if i==0:
        ax1.set_title('(g)Precipitation Intensity (MCS)',loc='left', pad=5,fontsize=16)
        ax1.set_yticklabels(['{:.1f}'.format(x) for x in ax1.get_yticks()],fontsize=13)
    elif i==1:
        ax1.set_title('(h)Precipitation Intensity (nonMCS)',loc='left', pad=5,fontsize=16)    
        ax1.set_yticklabels(['{:.2f}'.format(x) for x in ax1.get_yticks()],fontsize=13)


fig.savefig('NormalFigureSP4new.withSta_wrfncc.jpg',bbox_inches='tight',dpi=300)