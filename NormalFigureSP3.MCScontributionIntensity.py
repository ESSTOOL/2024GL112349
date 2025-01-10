# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:06:22 2024

@author: Dr.Yu
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

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

tracklist=['imerg','cmorph','gsmap','era5','wrf','wrfn']
labels=['IMERG','CMORPH','GSMaP','ERA5','WRF','WRF$_{nudging}$']
colors=['red','hotpink','firebrick','orange','cornflowerblue','green']
fig,axs=plt.subplots(nrows=1, ncols=1, figsize=(10,4))
fig.subplots_adjust(wspace=0.2,hspace=0.2)
ax1 = plt.subplot(111)
for i in range(6):
    data=np.loadtxt(tracklist[i]+'1yr.txt')[:,5112:]
    data[data>0]=1
    obs=np.loadtxt(tracklist[i]+'1yr_mcs.nc.mcsn.txt')[:,5112:]
    x=range(1,51)
    var=[]
    for j in range(50):
        themat=copy.deepcopy(obs)
        themat[themat<x[j]]=0
        themat[themat>0]=1
        fenmu=np.nansum(themat)
        themat[data==0]=0
        fenzi=np.nansum(themat)
        if fenmu>=10:
            var.append(fenzi/fenmu)
        else:
            var.append(np.nan)
        #if j>=45:
        #    print(tracklist[i],fenzi,fenmu)
    var=moving_average(np.array(var)*100)
    ax1.plot(x,var,color=colors[i],label=labels[i])

ax1.legend()
ax1.set_xlabel('Precipitation (mm/hr)')
ax1.set_ylabel('MCS-Associated Precipitation\'s Contribution (%)')

fig.savefig('NormalFigureSP3.MCScontributionIntensity.jpg',bbox_inches='tight',dpi=300)
            
        