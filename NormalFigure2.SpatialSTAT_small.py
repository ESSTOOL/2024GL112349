# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:18:18 2024

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
import cartopy

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def calc_grid_distance_area(lon,lat):
    """ Function to calculate grid parameters
        It uses haversine function to approximate distances
        It approximates the first row and column to the sencond
        because coordinates of grid cell center are assumed
        lat, lon: input coordinates(degrees) 2D [y,x] dimensions
        dx: distance (m)
        dy: distance (m)
        area: area of grid cell (m2)
        grid_distance: average grid distance over the domain (m)
    """
    dy = np.zeros(lon.shape)
    dx = np.zeros(lat.shape)

    dx[:,1:]=haversine(lon[:,1:],lat[:,1:],lon[:,:-1],lat[:,:-1])
    dy[1:,:]=haversine(lon[1:,:],lat[1:,:],lon[:-1,:],lat[:-1,:])

    dx[:,0] = dx[:,1]
    dy[0,:] = dy[1,:]
    
    dx = dx * 10**3
    dy = dy * 10**3

    area = dx*dy
    grid_distance = np.mean(np.append(dy[:, :, None], dx[:, :, None], axis=2))

    return dx,dy,area,grid_distance



data=xr.open_dataset('./imerg_202008.nc')
x=data.lon.values
y=data.lat.values
area=calc_grid_distance_area(x,y)[2][10:220,10:381]
# plt.pcolormesh(x,y,data.PR.values[0,:,:])
data.close()

tracklist=['imerg','cmorph','gsmap','era5','wrf','wrfn']
titles=['IMERG','CMORPH','GSMaP','ERA5','WRF','WRF$_{nudging}$']

fig,axs=plt.subplots(nrows=6, ncols=3, figsize=(16,21))
fig.subplots_adjust(wspace=0.1,hspace=0.275)


for i in range(6):
    ax1=plt.subplot(6,3,i+1,projection=ccrs.PlateCarree())
    locals()[tracklist[i]]=np.loadtxt('./supercompkl/'+tracklist[i]+'_warmmcsratio.txt')
    thispic=ax1.pcolormesh(x,y,locals()[tracklist[i]],cmap=cmaps.precip_11lev[[0,1,2,3,4,5,7,9]],alpha=1,vmin=0,vmax=0.8,transform=ccrs.PlateCarree())
    thispicdata=np.nansum(locals()[tracklist[i]][10:220,10:381]*area)/(np.nansum(area))
    #thispicdata=np.nanmean(locals()[tracklist[i]])
    ax1.set_title(titles[i]+':'+str(round(thispicdata,2)),loc='left', pad=5,fontsize=15)
    
    ax1.coastlines('50m', linewidth=0.8,zorder=1)
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='k', alpha=0.2, linestyle='--')
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator([100,105,110,115,120,125,130,135,140,145,150])
    gl.ylocator = mticker.FixedLocator([36,41,46,51,56])
    if i in [0,3]:
        gl.ylabel_style = {'size': 12}
    else:
        gl.ylabels_left = False
    if i in [3,4,5]:
        gl.xlabel_style = {'size': 12}
    else:
        gl.xlabels_bottom = False

ax1.text(9,88,'(a)',fontsize=20)

l = 0.92
b = 0.6625
w = 0.015
h = 0.2
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic, cax=cbar_ax,orientation='vertical',extend='max')
cb.set_label('Contribution of MCS-Associated Precpitation',fontsize=15)
cb.ax.tick_params(labelsize=11)


for i in range(6):
    ax1=plt.subplot(6,3,i+7,projection=ccrs.PlateCarree())
    locals()[tracklist[i]]=np.loadtxt('./supercompkl/'+tracklist[i]+'_warmmcsnum.txt')
    thispic=ax1.pcolormesh(x,y,locals()[tracklist[i]],cmap=cmaps.precip4_11lev[[0,1,2,3,4,5,6,7,9,10]],alpha=1,vmin=0,vmax=400,transform=ccrs.PlateCarree())
    thispicdata=np.nansum(locals()[tracklist[i]][10:220,10:381]*area)/(np.nansum(area))
    #thispicdata=np.nanmean(locals()[tracklist[i]])
    ax1.set_title(titles[i]+':'+str(int(thispicdata))+'hr',loc='left', pad=5,fontsize=15)
    
    ax1.coastlines('50m', linewidth=0.8,zorder=1)
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='k', alpha=0.2, linestyle='--')
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator([100,105,110,115,120,125,130,135,140,145,150])
    gl.ylocator = mticker.FixedLocator([36,41,46,51,56])
    if i in [0,3]:
        gl.ylabel_style = {'size': 12}
    else:
        gl.ylabels_left = False
    if i in [3,4,5]:
        gl.xlabel_style = {'size': 12}
    else:
        gl.xlabels_bottom = False

ax1.text(9,88,'(b)',fontsize=20)

l = 0.92
b = 0.4
w = 0.015
h = 0.2
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic, cax=cbar_ax,orientation='vertical',extend='max')
cb.set_label('MCS Frequency',fontsize=15)
cb.set_ticks(np.linspace(0,400,6))
cb.ax.tick_params(labelsize=11)



for i in range(6):
    ax1=plt.subplot(6,3,i+13,projection=ccrs.PlateCarree())
    locals()[tracklist[i]]=np.loadtxt('./supercompkl/'+tracklist[i]+'_warmmcspre.txt')
    thispic=ax1.pcolormesh(x,y,locals()[tracklist[i]],cmap=cmaps.precip3_16lev[[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16]],vmin=0,vmax=480,transform=ccrs.PlateCarree())
    thispicdata=np.nansum(locals()[tracklist[i]][10:220,10:381]*area)/(np.nansum(area))
    #thispicdata=np.nanmean(locals()[tracklist[i]])
    ax1.set_title(titles[i]+':'+str(int(thispicdata))+'mm',loc='left', pad=5,fontsize=15)
    
    ax1.coastlines('50m', linewidth=0.8,zorder=1)
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='k', alpha=0.2, linestyle='--')
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator([100,105,110,115,120,125,130,135,140,145,150])
    gl.ylocator = mticker.FixedLocator([36,41,46,51,56])
    if i in [0,3]:
        gl.ylabel_style = {'size': 12}
    else:
        gl.ylabels_left = False
    if i in [3,4,5]:
        gl.xlabel_style = {'size': 12}
    else:
        gl.xlabels_bottom = False

ax1.text(9,88,'(c)',fontsize=20)
        
l = 0.92
b = 0.14
w = 0.015
h = 0.2
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic, cax=cbar_ax,orientation='vertical',extend='max')
cb.set_label('MCS-Associated Precpitation (mm)',fontsize=15)
cb.set_ticks(np.linspace(0,480,7))
cb.ax.tick_params(labelsize=11)

fig.savefig('NormalFigure2.SpatialSTAT_small.jpg',bbox_inches='tight',dpi=300)
    