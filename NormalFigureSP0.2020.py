# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:44:39 2023

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
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
'''
data=xr.open_dataset('./obs_warmsum.nc')
pre=data.pre.values
lon=data.longitude.values
lat=data.latitude.values
lon_o,lat_o=np.meshgrid(lon,lat)
pre[pre<-90]=np.nan

# 加载形状文件
shapefile_path = "./shp/db.shp"
gdf = gpd.read_file(shapefile_path)

# 创建空的纬度和经度数据列表
lat_data_list = []
lon_data_list = []

# 遍历每个块
for i, block in gdf.iterrows():
    # 获取块的几何形状
    geometry = block.geometry

    # 创建当前块的纬度和经度数据矩阵
    lat_data = lat_o.copy()  # 复制原始纬度数据
    lon_data = lon_o.copy()  # 复制原始经度数据

    # 遍历每个格点
    for lat_index in range(360):
        for lon_index in range(640):
            # 获取当前格点的纬度和经度
            lat = lat_data[lat_index, lon_index]
            lon = lon_data[lat_index, lon_index]

            # 检查当前格点是否在块的几何形状内
            if not geometry.contains(Point(lon, lat)):
                # 如果不在内部，将格点的纬度和经度值改为nan
                lat_data[lat_index, lon_index] = np.nan
                lon_data[lat_index, lon_index] = np.nan

    # 将当前块的纬度和经度数据添加到列表中
    lat_data_list.append(lat_data)
    lon_data_list.append(lon_data)

# 打印每个块的纬度和经度数据
for i in range(len(lat_data_list)):
    lat_var_name = f"lat{i+1}"
    lon_var_name = f"lon{i+1}"
    locals()[lat_var_name] = lat_data_list[i]
    locals()[lon_var_name] = lon_data_list[i]

def calculateXulie(data,lat,lat_masked):
    xulie=[]
    for i in range(data.shape[0]):
        data[i,:,:][np.isnan(lat_masked)]=np.nan
        lat_cos = np.cos(np.deg2rad(lat))
        lat_cos = np.where(np.isnan(data[i,:,:]), np.nan, lat_cos)
        weighted_avg = np.nansum(data[i,:,:] * lat_cos) / np.nansum(lat_cos)
        xulie.append(weighted_avg)
    return(xulie)  

value=calculateXulie(pre,lat_o,lat1)
joblib.dump(value,'./OBSwarmsum_value.pkl')########1961-2022
'''
value=joblib.load('./OBSwarmsum_value.pkl')
import matplotlib.pyplot as plt
# plt.plot(range(len(datap)),datap)

fig,axs=plt.subplots(nrows=1, ncols=1, figsize=(10,6))
fig.subplots_adjust(wspace=0.2,hspace=0.2)

axs.plot(range(1961,2023),value,'k',lw=2)
axs.axvline(x=2020,ls='--',lw=1,color='r')
axs.set_xticks([1960,1970,1980,1990,2000,2010,2020])
axs.set_yticks([350,400,450,500,550])
axs.set_xticklabels([1960,1970,1980,1990,2000,2010,2020],fontsize=12)
axs.set_yticklabels([350,400,450,500,550],fontsize=12)
axs.set_xlabel('Year',fontsize=14)
axs.set_ylabel('Area-Weighted Accumulated Precipitation \n(Warm Season) (mm)',fontsize=14)

fig.savefig('NormalFigureSP0.2020.jpg',bbox_inches='tight',dpi=300)
