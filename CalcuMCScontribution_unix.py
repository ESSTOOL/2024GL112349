# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:44:36 2024

@author: Dr.Yu
"""

import xarray as xr
import numpy as np
import copy
#data=xr.open_dataset('./imerg_202007.nc')
#dataxx=data.PR.values
#dataaa=data.MCS_objects.values
tracklist=['imerg','cmorph','gsmap','era5','wrf','wrfn']
month=['202005','202006','202007','202008','202009']

for i in range(len(tracklist)):
    fenzimat=np.zeros((230,391))
    fenmumat=np.zeros((230,391))
    for j in range(len(month)):
        data=xr.open_dataset('./tracking_new/'+tracklist[i]+'_'+month[j]+'.nc')
        datapr=data.PR.values
        datamcs=data.mcs_mask.values
        datapr_copy=copy.deepcopy(datapr)
        
        fenmumat=fenmumat+np.nansum(datapr,axis=0)
        datapr_copy[datamcs<1]=0
        fenzimat=fenzimat+np.nansum(datapr_copy,axis=0)
    mcs_ratio=fenzimat/fenmumat
    np.savetxt('./supercompkl/'+tracklist[i]+'_warmmcsratio.txt',mcs_ratio)
        