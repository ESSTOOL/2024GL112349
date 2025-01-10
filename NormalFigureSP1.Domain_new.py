# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:31:35 2023

@author: Dr.Yu
"""
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib.patches as patches
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import  MultipleLocator
import matplotlib.ticker as mticker
import WRFDomainLibme
from matplotlib.patches import ConnectionPatch
import pandas as pd


fig,axs=plt.subplots(nrows=1, ncols=1, figsize=(10,6))
fig.subplots_adjust(wspace=0.2,hspace=0.15)

##############################

data=xr.open_dataset('./geo_em.d01big.nc')
#data=xr.open_dataset('./geo_em.d01.nc')
dem_lons=data.XLONG_M.values[0,:,:]
dem_lats=data.XLAT_M.values[0,:,:]
dem=data.HGT_M.values[0,:,:]

#from sklearn.externals import joblib
#joblib.dump(dem_lons,'xwrf.pkl')
#joblib.dump(dem_lats,'ywrf.pkl')


wrflonmax=144.49316
wrflonmin=105.506836
wrflatmax=57.202774
wrflatmin=34.398735
deltalatw=wrflatmax-wrflatmin
deltalonw=wrflonmax-wrflonmin
#box=[wrflonmin-7,wrflonmax+7,wrflatmin-5,wrflatmax+5]
box=[wrflonmin,wrflonmax,wrflatmin,wrflatmax]

cmap = matplotlib.cm.terrain
vmin = -200
vmax = 3800

ax1 = plt.subplot(111, projection=ccrs.PlateCarree())

thispic=ax1.pcolormesh(dem_lons, dem_lats, dem, cmap=cmap, vmin=vmin, vmax=vmax, alpha=1, transform=ccrs.PlateCarree(), zorder=0)
stainfo=pd.read_csv('stationwithheight.txt.csv')
ax1.scatter(stainfo['Lon'],stainfo['Lat'],color='k',marker='+',linewidth=1,s=20,alpha=1, transform=ccrs.PlateCarree(), label='Gauge Observations',zorder=3)

ax1.set_extent(box, crs=ccrs.PlateCarree())
ax1.legend(fontsize=14)
# decorations
ax1.coastlines('50m', linewidth=0.8,zorder=1)
ax1.add_feature(cartopy.feature.OCEAN, edgecolor='k', facecolor='lightblue', zorder=2)
#ax1.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='orange', zorder=1)


gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='k', alpha=0.2, linestyle='--')
gl.xlabels_top = False  # 关闭顶端的经纬度标签
gl.ylabels_right = False  # 关闭右侧的经纬度标签
gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
gl.xlocator = mticker.FixedLocator([90,95,100,105,110,115,120,125,130,135,140,145,150,155,160])
gl.ylocator = mticker.FixedLocator([26,31,36,41,46,51,56,61,66])
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

#ax1.add_patch(matplotlib.patches.Rectangle((wrflonmin, wrflatmin), deltalonw,deltalatw, fill=None, lw=2, edgecolor='k', zorder=2))
shpfn = './shp/db2.shp'
reader = shpreader.Reader(shpfn)
myshp = cfeature.ShapelyFeature(reader.geometries(), crs = ccrs.PlateCarree(), edgecolor = 'orange', facecolor = 'None')
ax1.add_feature(myshp, linewidth=3)
#ax1.add_patch(matplotlib.patches.Rectangle((wrflonmin2, wrflatmin2), deltalonw2, deltalatw2, fill=None, lw=2, edgecolor='r', zorder=2))
# ax1.add_patch(matplotlib.patches.Rectangle((wrflonmin1, wrflatmin1), deltalonw1, deltalatw1, fill=None, lw=2, edgecolor='white', zorder=2))

#ax1.text(106.5,55.5,'D1',color='k',fontsize=12)
# ax1.text(100,58,'D2',color='r',fontsize=12)
# ax1.text(93.5,61,'D3',color='k',fontsize=12)


# data1.close()
# data2.close()
data.close()


l = 0.175
b = 0.03
w = 0.7
h = 0.0175
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb=plt.colorbar(thispic, cax=cbar_ax,extend='both',orientation='horizontal')
cb.set_ticks(np.arange(-200, vmax+1, 400))
cb.ax.tick_params(labelsize=12)
cb.set_label('Elevation (m)',fontsize=16)
cb.ax.tick_params(labelsize=13)

fig.savefig('NormalFigureSP1.Domain_new.jpg',bbox_inches='tight',dpi=300)