# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:45:27 2024

@author: Dr.Yu
"""

from sklearn.externals import joblib
import numpy as np

#ddf=joblib.load('./era5_201911.nc_grMCSs.pkl')
#ddf2=joblib.load('./imerg_202008.nc_grMCSs.pkl')

tracklist=['imerg','cmorph','gsmap','era5','wrf','wrfn']
month=['201910','201911','201912','202001','202002','202003','202004','202005','202006','202007','202008','202009']

for i in range(len(tracklist)):
    locals()[tracklist[i]+'GRMCS']=np.full((12,5),np.nan)
    for j in range(len(month)):
        mcsdata=joblib.load('./tracking/'+tracklist[i]+'_'+month[j]+'.nc_grMCSs.pkl')
        if (mcsdata is None):
            locals()[tracklist[i]+'GRMCS'][j,0]=0
        else:
            locals()[tracklist[i]+'GRMCS'][j,0]=len(mcsdata.keys())
            times=[]
            sizes=[]
            means=[]
            maxs=[]
            for k in range(len(mcsdata.keys())):
                times.append(len(mcsdata[list(mcsdata.keys())[k]]['times']))
                sizes.append(np.nanmean(mcsdata[list(mcsdata.keys())[k]]['size']))
                means.append(np.nanmean(mcsdata[list(mcsdata.keys())[k]]['mean']))
                maxs.append(np.nanmean(mcsdata[list(mcsdata.keys())[k]]['max']))
                
            if (len(mcsdata.keys())>=10):
                locals()[tracklist[i]+'GRMCS'][j,1]=np.nanmedian(times)
                locals()[tracklist[i]+'GRMCS'][j,2]=np.nanmedian(sizes)
                locals()[tracklist[i]+'GRMCS'][j,3]=np.nanmedian(means)
                locals()[tracklist[i]+'GRMCS'][j,4]=np.nanmedian(maxs)
            else:
                locals()[tracklist[i]+'GRMCS'][j,1]=9999
                locals()[tracklist[i]+'GRMCS'][j,2]=9999
                locals()[tracklist[i]+'GRMCS'][j,3]=9999
                locals()[tracklist[i]+'GRMCS'][j,4]=9999              
    joblib.dump(locals()[tracklist[i]+'GRMCS'],tracklist[i]+'_statn.pkl')