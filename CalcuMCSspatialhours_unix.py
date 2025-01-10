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
    mcscountmat=np.zeros((230,391))
    for j in range(len(month)):
        data=xr.open_dataset('./tracking_new/'+tracklist[i]+'_'+month[j]+'.nc')
        datamcs=data.mcs_mask.values
        datamcs[datamcs>=1]=1
        mcscountmat=mcscountmat+np.sum(datamcs,axis=0)
    np.savetxt('./supercompkl/'+tracklist[i]+'_warmmcsnum.txt',mcscountmat)
        