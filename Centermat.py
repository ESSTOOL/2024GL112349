# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:32:30 2024

@author: Dr.Yu
"""

from sklearn.externals import joblib
import numpy as np
import xarray as xr
import copy

# datamcs=joblib.load('imerg_202008.nc_grMCSs.pkl')
# datanc=xr.open_dataset('imerg_202008.nc')

tracklist=['imerg','cmorph','gsmap','era5','wrf','wrfn']
month=['202005','202006','202007','202008','202009']

for i in range(len(tracklist)):
    godmat_PR=np.full((200,61,61),np.nan)
    godmat_BT=np.full((200,61,61),np.nan)
    calculator=0
    for j in range(len(month)):
        datamcs=joblib.load('./tracking_new/'+tracklist[i]+'_'+month[j]+'.nc_grMCSs.pkl')
        datanc=xr.open_dataset('./tracking_new/'+tracklist[i]+'_'+month[j]+'.nc')
        datamcspr=copy.deepcopy(datanc.PR.values)
        datamcspr[datanc.mcs_mask.values==0]=np.nan
        datamcsbt=copy.deepcopy(datanc.BT.values)
        datamcsbt[datanc.mcs_mask.values==0]=np.nan
        nctime=datanc.time.values
        if (datamcs is None):
            print('oh shit!')
        else:
            for k in range(len(datamcs.keys())):
                mcsno=int(list(datamcs.keys())[k])
                maxprecip=datamcs[list(datamcs.keys())[k]]['max']
                maxplace=np.where(maxprecip==np.nanmax(maxprecip))
                amcstime=datamcs[list(datamcs.keys())[k]]['times'][maxplace][0]
                timeplace=np.where(nctime==amcstime)[0][0]
                theprdata=copy.deepcopy(datanc.PR.values[timeplace,:,:])
                themcsdata=copy.deepcopy(datanc.mcs_mask.values[timeplace,:,:])
                theprdata[themcsdata!=mcsno]=np.nan
                locplace=np.where(theprdata==np.nanmax(maxprecip))
                thelat=datanc.lat.values[locplace[0][0],locplace[1][0]]
                thelon=datanc.lon.values[locplace[0][0],locplace[1][0]]
                if (((thelon>=108.44999) and (thelon<=141.45) and (thelat>=37.34999) and (thelat<=54.25))==True):
                    godmat_PR[calculator,:,:]=datamcspr[timeplace,(locplace[0][0]-30):(locplace[0][0]+31),(locplace[1][0]-30):(locplace[1][0]+31)]
                    godmat_BT[calculator,:,:]=datamcsbt[timeplace,(locplace[0][0]-30):(locplace[0][0]+31),(locplace[1][0]-30):(locplace[1][0]+31)]
                    calculator=calculator+1
                    
    joblib.dump(godmat_PR,tracklist[i]+'_shapemat_PRmax.pkl')
    joblib.dump(godmat_BT,tracklist[i]+'_shapemat_BTmax.pkl')